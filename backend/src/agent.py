import logging
import json
import os
from datetime import datetime
from typing import Optional, List
import re 

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


# --- INITIAL ORDER STATE ---
INITIAL_ORDER_STATE = {
    "drinkType": None,
    "size": None,
    "milk": None,
    # 'extras' is required to be filled (list must be non-empty) before completion
    "extras": [], 
    "name": None
}
# ---------------------------


class BaristaAgent(Agent):
    def __init__(self) -> None:
        barista_instructions = (
            "You are a friendly and efficient coffee shop barista at 'The Falcon Brew'. "
            "Your primary goal is to take a customer's order by asking one clarifying question at a time "
            "until the entire order state is filled. "
            # UPDATED: 'extras' is now listed as a required field for the LLM to prompt for it.
            "The required fields are: drinkType, size, milk, name, and at least one item in extras. "
            "Use the 'update_coffee_order' tool whenever the user provides a relevant piece of information. "
            "Always ask the user for the next missing required field in the order state. If all other fields are filled, ask the customer what extras they would like (e.g., 'extra shot,' 'vanilla syrup,' etc.)."
        )

        # State flag to prevent race conditions and early completion
        self._is_order_complete = False 
        
        super().__init__(instructions=barista_instructions)
        
        self.order_state = INITIAL_ORDER_STATE.copy()
        
        self.update_tools([self.update_order_tool])

    @function_tool(
        name="update_coffee_order",
        description="Call this function to update any missing field in the current coffee order. Use it only when the user explicitly provides a value for a field. Never guess. The 'extras' field can be updated multiple times.",
    )
    async def update_order_tool(
        self,
        context: RunContext,
        # FIX: Using Optional to allow LLM to pass None/null without validation error
        drinkType: Optional[str] = None, 
        size: Optional[str] = None,
        milk: Optional[str] = None,
        extras: Optional[List[str]] = None, 
        name: Optional[str] = None,
    ):
        """Used to update the customer's coffee order state. Call this once for every piece of information the user provides."""

        if self._is_order_complete:
            return "Order is already complete and saved. Thank the customer and confirm their order."
        
        # Helper function to treat None and empty string as invalid/missing input
        def is_valid_input(value):
            return value is not None and value != ""

        # Update the local state only with valid (non-None and non-empty string) arguments
        if is_valid_input(drinkType):
            self.order_state["drinkType"] = drinkType
        if is_valid_input(size):
            self.order_state["size"] = size
        if is_valid_input(milk):
            self.order_state["milk"] = milk
        if is_valid_input(name):
            self.order_state["name"] = name
        
        # Handle 'extras' (accumulating items and avoiding duplicates)
        if extras is not None and isinstance(extras, list):
            for item in extras:
                if item not in self.order_state["extras"]:
                    self.order_state["extras"].append(item)


        # REQUIRED FIELDS CHECK
        required_fields = ["drinkType", "size", "milk", "name"]
        
        # Identify missing fields for the simple string/None required fields (FIX: Check for empty string)
        missing_fields = [k for k in required_fields if self.order_state[k] is None or self.order_state[k] == ""]

        # NEW/FIX: Check the 'extras' requirement (list must have at least one item)
        if not self.order_state["extras"]:
            # If the list is empty, treat 'extras' as missing
            missing_fields.append("extras (e.g., extra shot, syrup)")

        if not missing_fields:
            # All required fields are filled, including at least one extra
            self._is_order_complete = True
            
            await self._save_order_and_summarize(context)
            
            return "Order is complete and saved. Thank the customer and confirm their order."
        else:
            # Tell the LLM what is still missing so it can ask the next question
            return f"Order state updated. The following is still missing: {', '.join(missing_fields)}. Ask the user for the next piece of missing required information."

    async def _save_order_and_summarize(self, context: RunContext):
        """Saves the final order state to a JSON file and sends a summary to the user."""
        
        ORDERS_DIR = "orders"
        if not os.path.exists(ORDERS_DIR):
            try:
                os.makedirs(ORDERS_DIR)
            except OSError as e:
                logger.error(f"Failed to create orders directory {ORDERS_DIR}: {e}")
                return

        # Robust filename generation
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        raw_name = self.order_state.get("name", "anon_order")
        # Sanitize the name: replace non-alphanumeric characters (except - and _)
        sanitized_name = raw_name.replace(" ", "_").lower()
        order_name = re.sub(r'[^\w\-]', '', sanitized_name) 

        filename = os.path.join(ORDERS_DIR, f"order_{order_name}_{current_time}.json")
        
        # FIX: Added Try/Except Block for File Writing
        try:
            # Write the JSON file
            with open(filename, 'w') as f:
                json.dump(self.order_state, f, indent=4)
                
            # Print confirmation to the terminal 
            print("=" * 50)
            print(f"🎉 DAY 2 PRIMARY GOAL COMPLETE! Order Saved to: {filename}")
            print(json.dumps(self.order_state, indent=4))
            print("=" * 50)

        except Exception as e:
            logger.error(f"Failed to save order to file {filename}: {e}")
            
            
        # Send a final summary to the user via TTS
        customer_name = self.order_state.get("name", "there")
        extras_list = f" with {' and '.join(self.order_state['extras'])}" if self.order_state['extras'] else ''
        summary = f"Got it, {customer_name}! Your order has been placed. That's one {self.order_state['size']} {self.order_state['drinkType']} with {self.order_state['milk']}{extras_list}. We'll have that ready shortly."
        await context.session.say(summary)


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
        agent=BaristaAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))