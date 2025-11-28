import logging
import json
import os
from datetime import datetime
import uuid
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# --- Configuration and Data Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CATALOG_FILE = os.path.join(DATA_DIR, "catalog.json")
RECIPES_FILE = os.path.join(DATA_DIR, "recipes.json")
# Orders will be saved one directory up from the 'src' folder, e.g., 'backend/orders'
ORDERS_DIR = os.path.join(os.path.dirname(__file__), "..", "orders")
TAX_RATE = 0.08  # 8% sales tax

def load_json_file(filepath: str) -> List[Dict[str, Any]]:
    """Helper function to load catalog or recipes data."""
    if not os.path.exists(filepath):
        # Create data directory if it doesn't exist to avoid crashing
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        print(f"Error: File not found at {filepath}. Please create it.")
        return []
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}. Check formatting.")
        return []


class Assistant(Agent):
    def __init__(self) -> None:
        # Load data and initialize cart state in the agent instance
        self.catalog = load_json_file(CATALOG_FILE)
        self.recipes = load_json_file(RECIPES_FILE)
        self.catalog_map = {item['name'].lower(): item for item in self.catalog}
        self.recipes_map = {recipe['recipe_name'].lower(): recipe for recipe in self.recipes}
        self.cart: List[Dict[str, Any]] = []

        # Ensure orders directory exists
        os.makedirs(ORDERS_DIR, exist_ok=True)

        super().__init__(
            instructions="""You are **QuickMart's friendly food and grocery ordering assistant**. 
            Your primary job is to help the user order items from our catalog. You must use the provided tools to manage the user's cart. 
            Your responses should be encouraging, concise, and focused on helping the user shop. 
            
            **Key instructions:**
            1. Always confirm back to the user what you have added or removed from their cart.
            2. When the user says they are done, for example: "Place my order", "I'm done", or "That's all", you **must** call the `place_order` tool.
            3. If the user asks what they need for a meal (e.g., "ingredients for a sandwich"), use the `add_ingredients_for_recipe` tool.
            4. Only use your general knowledge if the tools cannot answer a non-ordering question.
            Your responses are concise, to the point, and without any complex formatting including emojis, asterisks, or other weird symbols.""",
        )
        logger.info("QuickMart Agent Initialized. Catalog and Recipes Loaded.")
    
    # --- Internal Helper Methods (Synchronous Logic) ---

    def _get_item_details(self, item_name: str) -> Optional[Dict[str, Any]]:
        """Looks up item details by name in the catalog."""
        return self.catalog_map.get(item_name.lower())

    def _calculate_total(self) -> Dict[str, float]:
        """Calculates the subtotal, tax, and grand total of the current cart."""
        subtotal = 0.0
        for item in self.cart:
            subtotal += item['line_total_usd']

        tax = subtotal * TAX_RATE
        total = subtotal + tax

        return {
            "subtotal_usd": round(subtotal, 2),
            "tax_usd": round(tax, 2),
            "order_total_usd": round(total, 2)
        }
    
    # NEW SYNCHRONOUS HELPER for adding item
    def _add_item_to_cart_sync(self, item_name: str, quantity: int) -> str:
        """Internal synchronous function to add an item to the cart."""
        if quantity <= 0:
            return f"Please specify a quantity greater than zero to add {item_name}."

        item_details = self._get_item_details(item_name)
        if not item_details:
            return f"I'm sorry, I couldn't find '{item_name}' in our catalog. Could you please specify a different item?"

        # Prepare cart item details
        price = item_details['price_usd']
        cart_item_template = {
            "item_id": item_details['item_id'],
            "name": item_details['name'],
            "price_usd": price,
            "unit": item_details.get('unit', 'unit')
        }

        # Check if item is already in cart to update quantity
        existing_item = next((item for item in self.cart if item['item_id'] == cart_item_template['item_id']), None)
        
        if existing_item:
            existing_item['quantity'] += quantity
            existing_item['line_total_usd'] = round(price * existing_item['quantity'], 2)
            return f"Updated: Added {quantity} more of {item_name}. You now have {existing_item['quantity']} total in the cart."
        else:
            new_item = cart_item_template.copy()
            new_item['quantity'] = quantity
            new_item['line_total_usd'] = round(price * quantity, 2)
            self.cart.append(new_item)
            return f"Added {quantity} of {item_name} to your cart. Is there anything else you need?"

    # NEW SYNCHRONOUS HELPER for removing item
    def _remove_item_from_cart_sync(self, item_name: str, quantity: Optional[int] = None) -> str:
        """Internal synchronous function to remove an item from the cart."""
        item_details = self._get_item_details(item_name)
        if not item_details:
            return f"I couldn't find '{item_name}' in your cart or the catalog to remove it."

        item_id = item_details['item_id']
        existing_item = next((item for item in self.cart if item['item_id'] == item_id), None)

        if not existing_item:
            return f"'{item_name}' is not currently in your cart."

        if quantity is None or quantity >= existing_item['quantity']:
            self.cart = [item for item in self.cart if item['item_id'] != item_id]
            return f"Removed all units of {item_name} from your cart."
        
        # Reduce quantity
        if quantity <= 0:
            return f"Please specify a positive quantity to remove from {item_name}."
            
        existing_item['quantity'] -= quantity
        existing_item['line_total_usd'] = round(existing_item['price_usd'] * existing_item['quantity'], 2)
        return f"Removed {quantity} of {item_name}. You now have {existing_item['quantity']} remaining. Anything else?"


    # --- Agent Tools (Callable by LLM) ---

    @function_tool
    async def add_item_to_cart(self, context: RunContext, item_name: str, quantity: int) -> str:
        """
        Adds a specific item and quantity to the cart, or updates the quantity if the item is already present.
        
        Args:
            item_name: The name of the item to add (e.g., "Large Eggs", "Whole Wheat Bread").
            quantity: The number of units to add (must be a positive integer).
        """
        # Call the synchronous helper
        return self._add_item_to_cart_sync(item_name, quantity)


    @function_tool
    async def remove_item_from_cart(self, context: RunContext, item_name: str, quantity: Optional[int] = None) -> str:
        """
        Removes an item from the cart or reduces its quantity.

        Args:
            item_name: The name of the item to remove.
            quantity: The amount to remove. If None or greater than current quantity, removes all of the item.
        """
        # Call the synchronous helper
        return self._remove_item_from_cart_sync(item_name, quantity)


    @function_tool
    async def add_ingredients_for_recipe(self, context: RunContext, recipe_name: str) -> str:
        """
        Adds multiple items for a common meal/recipe (e.g., a 'peanut butter sandwich') to the cart.
        
        Args:
            recipe_name: The name of the recipe (e.g., "pasta dinner", "tacos").
        """
        recipe = self.recipes_map.get(recipe_name.lower())
        if not recipe:
            return f"I'm sorry, I don't have a recipe for '{recipe_name}' in my system. I can still add items one by one."

        for required in recipe['required_items']:
            item_name = required['item_name']
            quantity = required['quantity']
            
            # *** FIX APPLIED HERE: CALLING THE SYNCHRONOUS HELPER ***
            self._add_item_to_cart_sync(item_name, quantity)

        return recipe['confirmation_message']

    @function_tool
    async def list_cart_contents(self, context: RunContext) -> str:
        """
        Lists the contents of the current cart, including the estimated subtotal.

        Returns:
            A formatted string summary of the cart.
        """
        if not self.cart:
            return "Your cart is empty. What would you like to order first?"

        totals = self._calculate_total()
        
        response = "Here's what's in your cart:\n"
        for item in self.cart:
            response += f"- {item['quantity']} x {item['name']} (Total: ${item['line_total_usd']:.2f})\n"

        response += f"\nYour current estimated subtotal is ${totals['subtotal_usd']:.2f}."
        return response

    @function_tool
    async def place_order(self, context: RunContext, customer_name: str = "Customer", address: str = "Unknown Address") -> str:
        """
        Finalizes the order, calculates the total, saves it to a JSON file, and clears the cart.
        This must be called when the user indicates they are done ordering (e.g., "Place my order").

        Args:
            customer_name: The customer's name for the order record (can be generic if unknown).
            address: The delivery address or a simple customer note.
        """
        if not self.cart:
            return "I can't place an order because your cart is empty. Please add some items first."

        totals = self._calculate_total()
        order_id = f"QM-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
        timestamp = datetime.now().isoformat()

        # Construct the final order object
        order_data = {
            "order_id": order_id,
            "timestamp": timestamp,
            "customer_info": {
                "name": customer_name,
                "address": address
            },
            "items": self.cart,
            "subtotal_usd": totals['subtotal_usd'],
            "tax_usd": totals['tax_usd'],
            "order_total_usd": totals['order_total_usd'],
            "status": "received"
        }

        # Save the order to a JSON file
        filename = os.path.join(ORDERS_DIR, f"order_{customer_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        try:
            with open(filename, 'w') as f:
                json.dump(order_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving order to file: {e}")
            return f"I encountered an error while trying to save your order, but your total is ${totals['order_total_usd']:.2f}. Your order has been placed but may not be logged correctly."

        # Clear the cart after placing the order
        self.cart = []
        
        # Final confirmation to the user
        confirmation = (
            f"Thank you for your order, {customer_name}! "
            f"Your order ID is {order_id}. "
            f"The final total is ${totals['order_total_usd']:.2f} (including ${totals['tax_usd']:.2f} in tax). "
            f"Your order has been placed and saved successfully!"
        )
        return confirmation


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(
                model="gemini-2.5-flash",
            ),
        tts=murf.TTS(
                voice="en-US-matthew", 
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # Metrics collection, to measure pipeline performance
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))