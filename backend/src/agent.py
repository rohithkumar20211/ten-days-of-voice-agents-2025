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

# --- Configuration and Data Paths (Re-introduced from Day 7) ---
# Assuming 'data' is a sibling to 'src'
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CATALOG_FILE = os.path.join(DATA_DIR, "catalog.json")
ORDERS_DIR = os.path.join(os.path.dirname(__file__), "..", "orders")
TAX_RATE = 0.08  # 8% sales tax

def load_json_file(filepath: str) -> List[Dict[str, Any]]:
    """Helper function to load catalog data."""
    if not os.path.exists(filepath):
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
        # Load data and initialize cart state
        self.catalog = load_json_file(CATALOG_FILE)
        self.catalog_map = {item['name'].lower(): item for item in self.catalog}
        self.cart: List[Dict[str, Any]] = []

        # Ensure orders directory exists
        os.makedirs(ORDERS_DIR, exist_ok=True)

        super().__init__(
            # Day 9 Primary Goal: E-commerce Agent Persona (ACP-inspired)
            instructions="""You are the **E-commerce Shopping Assistant** for a high-end online store. 
            Your role is to efficiently guide the user through browsing the catalog, adding items to their order, and finalizing the purchase.
            
            **Key Instructions:**
            1. Your tone must be professional, concise, and helpful.
            2. You **must** use the provided tools (`browse_catalog`, `add_item_to_order`, `place_order`) to fulfill the user's shopping intent. Do not fabricate product information.
            3. Always confirm the final order details and total when the user decides to checkout.
            4. Your responses must be concise, to the point, and without complex formatting.
            
            Initial Greeting: Start by welcoming the user and asking what kind of product they are looking for, or if they want to browse the catalog.
            """,
        )
        logger.info("E-commerce Agent Initialized. Catalog Loaded.")
    
    # --- Internal Helper Methods ---

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

    # --- ACP-Inspired Merchant Tools ---

    @function_tool
    async def browse_catalog(self, context: RunContext, query: Optional[str] = None) -> str:
        """
        Retrieves a summary of the available products. Use this to help the user shop.

        Args:
            query: An optional search term to filter products by name or category (e.g., "bread" or "prepared food").
        """
        if not self.catalog:
            return "The product catalog is currently empty."
        
        products = self.catalog
        if query:
            query = query.lower()
            products = [
                item for item in self.catalog 
                if query in item['name'].lower() or query in item['category'].lower()
            ]

        if not products:
            return f"I found no products matching your query '{query}'. Please try a different search term or category."
            
        summary = "Available products:\n"
        for item in products[:5]: # Limit output to 5 for conciseness
            summary += f"- {item['name']} (Category: {item['category']}, Price: ${item['price_usd']:.2f})\n"
        
        if len(products) > 5:
             summary += f"And {len(products) - 5} more items. What item would you like to add to your order?"
             
        return summary


    @function_tool
    async def add_item_to_order(self, context: RunContext, item_name: str, quantity: int) -> str:
        """
        Adds a specific item and quantity to the current order.

        Args:
            item_name: The exact name of the product to add (e.g., "Large Eggs").
            quantity: The number of units to add (must be a positive integer).
        """
        if quantity <= 0:
            return f"The quantity for '{item_name}' must be greater than zero."

        item_details = self._get_item_details(item_name)
        if not item_details:
            return f"I couldn't find '{item_name}' in the catalog. Please use the exact product name or try browsing the catalog."

        # Prepare item details
        price = item_details['price_usd']
        cart_item_template = {
            "item_id": item_details['item_id'],
            "name": item_details['name'],
            "price_usd": price,
            "currency": "USD",
        }

        # Check if item is already in cart to update quantity (Standard Cart Logic)
        existing_item = next((item for item in self.cart if item['item_id'] == cart_item_template['item_id']), None)
        
        if existing_item:
            existing_item['quantity'] += quantity
            existing_item['line_total_usd'] = round(price * existing_item['quantity'], 2)
            return f"Updated: Added {quantity} more of {item_name}. You now have {existing_item['quantity']} total in your order."
        else:
            new_item = cart_item_template.copy()
            new_item['quantity'] = quantity
            new_item['line_total_usd'] = round(price * quantity, 2)
            self.cart.append(new_item)
            return f"Added {quantity} of {item_name} to your order for ${new_item['line_total_usd']:.2f}. Would you like to add anything else or proceed to checkout?"

    @function_tool
    async def place_order(self, context: RunContext, customer_name: str = "Customer", address_note: str = "Delivery to file") -> str:
        """
        Finalizes the order, calculates the total, saves it to a JSON file (simulating order placement), and clears the cart.
        This must be called when the user indicates they are done shopping (e.g., "Checkout" or "Place my order").

        Args:
            customer_name: The customer's name for the order record.
            address_note: A simple note about the delivery or transaction.
        """
        if not self.cart:
            return "I cannot place an order because your cart is empty. Please add items first."

        totals = self._calculate_total()
        order_id = f"ACP-QM-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
        timestamp = datetime.now().isoformat()

        # Construct the final order object (ACP-inspired structure)
        order_data = {
            "order_id": order_id,
            "timestamp": timestamp,
            "customer_info": {
                "name": customer_name,
                "address_note": address_note
            },
            "items": self.cart,
            "subtotal_usd": totals['subtotal_usd'],
            "tax_usd": totals['tax_usd'],
            "order_total_usd": totals['order_total_usd'],
            "currency": "USD",
            "status": "order_placed"
        }

        # Save the order to a JSON file (Task 5)
        filename = os.path.join(ORDERS_DIR, f"acp_order_{customer_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        try:
            with open(filename, 'w') as f:
                json.dump(order_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving order to file: {e}")
            return f"I encountered an error while trying to save your order, but your total is ${totals['order_total_usd']:.2f}. The order was placed but may not be logged correctly."

        # Clear the cart after placing the order
        self.cart = []
        
        # Final confirmation to the user
        confirmation = (
            f"Your order (ID: {order_id}) has been placed successfully. "
            f"The final total is ${totals['order_total_usd']:.2f}. "
            f"Thank you for shopping with us!"
        )
        return confirmation


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
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

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))