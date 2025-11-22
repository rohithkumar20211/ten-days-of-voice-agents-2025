import logging
import json
import os
from datetime import datetime

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


# --- DAY 2: INITIAL ORDER STATE ---
INITIAL_ORDER_STATE = {
    "drinkType": None,
    "size": None,
    "milk": None,
    "extras": [],
    "name": None
}
# ----------------------------------


class BaristaAgent(Agent):
    def __init__(self) -> None:
        # Define the Barista persona and task instructions for the LLM
        barista_instructions = (
            "You are a friendly and efficient coffee shop barista at 'The Falcon Brew'. "
            "Your primary goal is to take a customer's order by asking one clarifying question at a time "
            "until the entire order state is filled. "
            "The required fields are: drinkType, size, milk, and name. Extras are optional. "
            "Use the 'update_coffee_order' tool whenever the user provides a relevant piece of information. "
            "Always ask the user for the next missing required field in the order state."
        )

        super().__init__(instructions=barista_instructions)
        
        # Initialize the order state
        self.order_state = INITIAL_ORDER_STATE.copy()
        
        # Make the order tool available to the LLM
        self.update_tools([self.update_order_tool])

    @function_tool(
        name="update_coffee_order",
        description="Call this function to update any missing field in the current coffee order. Use it only when the user explicitly provides a value for a field. Never guess. The 'extras' field can be updated multiple times.",
    )
    async def update_order_tool(
        self,
        context: RunContext,
        drinkType: str = None,
        size: str = None,
        milk: str = None,
        extras: list[str] = None,
        name: str = None,
    ):
        """Used to update the customer's coffee order state. Call this once for every piece of information the user provides."""
        
        # 1. Update the local state with any non-None arguments
        if drinkType is not None:
            self.order_state["drinkType"] = drinkType
        if size is not None:
            self.order_state["size"] = size
        if milk is not None:
            self.order_state["milk"] = milk
        if name is not None:
            self.order_state["name"] = name
        
        # Handle 'extras' as a list that can accumulate items
        if extras is not None and isinstance(extras, list):
            self.order_state["extras"].extend(extras)

        # 2. Check if all REQUIRED fields are filled (excluding 'extras')
        required_fields = ["drinkType", "size", "milk", "name"]
        missing_fields = [k for k in required_fields if self.order_state[k] is None]

        if not missing_fields:
            # All required fields are filled, complete the order
            await self._save_order_and_summarize(context)
            
            # This message is sent to the LLM to confirm completion and stop questioning
            return "Order is complete and saved. Thank the customer and confirm their order."
        else:
            # Tell the LLM what is still missing so it can ask the next question
            return f"Order state updated. The following is still missing: {', '.join(missing_fields)}. Ask the user for the next piece of missing required information."

    async def _save_order_and_summarize(self, context: RunContext):
        """Saves the final order state to a JSON file and sends a summary to the user."""
        
        # Create a directory to store orders if it doesn't exist
        ORDERS_DIR = "orders"
        if not os.path.exists(ORDERS_DIR):
            os.makedirs(ORDERS_DIR)
            
        # Format the filename
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        order_name = self.order_state.get("name", "anon_order").replace(" ", "_")
        filename = os.path.join(ORDERS_DIR, f"order_{order_name}_{current_time}.json")
        
        # Write the JSON file
        with open(filename, 'w') as f:
            json.dump(self.order_state, f, indent=4)
            
        # Print confirmation to the terminal (for video recording proof)
        print("=" * 50)
        print(f"🎉 DAY 2 PRIMARY GOAL COMPLETE! Order Saved to: {filename}")
        print(json.dumps(self.order_state, indent=4))
        print("=" * 50)
        
        # Optional: Send a final summary to the user via TTS
        # This gives the agent a final response to say out loud before hanging up
        extras_list = f" and {' and '.join(self.order_state['extras'])}" if self.order_state['extras'] else ''
        summary = f"Got it, {self.order_state['name']}! Your order has been placed. That's one {self.order_state['size']} {self.order_state['drinkType']} with {self.order_state['milk']}{extras_list}. We'll have that ready shortly."
        await context.session.say(summary)


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using Murf, Deepgram, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        stt=deepgram.STT(model="nova-3"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        llm=google.LLM(
                model="gemini-2.5-flash",
            ),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        tts=murf.TTS(
                voice="en-US-matthew", 
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
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
        agent=BaristaAgent(), # <--- Using the new BaristaAgent
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
