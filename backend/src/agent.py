import logging
import json
import os
from datetime import datetime, timedelta
from typing import List, Optional

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

# --- DAY 3: FILE PATH & DATE MOCKING STATE ---
WELLNESS_LOG_FILE = "wellness_log.json"
# Global state to hold the mock date for the current session.
# We will manually advance this between calls to simulate consecutive days.
MOCK_DATE = datetime.now().date() + timedelta(days = 1)# Simulate Day 3
# ----------------------------------------------


def load_wellness_history() -> List[dict]:
    """Loads the past check-ins from the JSON file."""
    if os.path.exists(WELLNESS_LOG_FILE):
        with open(WELLNESS_LOG_FILE, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print(f"WARNING: {WELLNESS_LOG_FILE} is corrupted or empty. Starting fresh.")
                return []
    return []


def save_new_entry(new_entry: dict):
    """Appends a new entry to the wellness log file."""
    history = load_wellness_history()
    
    # CRITICAL: Add the current MOCK_DATE to the entry before saving
    new_entry["date"] = MOCK_DATE.strftime("%Y-%m-%d") 
    
    history.append(new_entry)
    with open(WELLNESS_LOG_FILE, 'w') as f:
        json.dump(history, f, indent=4)


class WellnessAgent(Agent):
    def __init__(self) -> None:
        # Load historical data at agent startup
        self.history = load_wellness_history()
        self.last_checkin_summary = self._get_last_summary()
        
        # 1. Define the system prompt with persona and rules
        base_instructions = (
            "You are a supportive, realistic, and grounded Health & Wellness Voice Companion. "
            "Your goal is to conduct a short daily check-in with the user, focusing on mood and goals. "
            "NEVER offer diagnosis or medical advice. Keep advice small, actionable, and non-medical. "
            "Your conversation flow must be: "
            "1. Acknowledge user's history (if provided in context). "
            "2. Ask about current mood and energy. "
            "3. Ask about 1-3 simple, practical objectives for the day. "
            "4. Offer one simple, grounded piece of advice. "
            "5. Call the 'complete_checkin' tool to summarize and save the log."
        )
        
        # 4. Use past data to inform the conversation
        if self.last_checkin_summary:
            base_instructions += f"\n\nContext on Past Check-in: The user's last session was on {self.history[-1]['date']}. They reported: {self.last_checkin_summary}. Use this to ask how their current state compares or if they followed through on a previous goal."
        
        super().__init__(instructions=base_instructions)
        
        # Add the tool to the agent's available tools
        self.update_tools([self.complete_checkin])

    def _get_last_summary(self) -> Optional[str]:
        """Returns a summary of the most recent check-in, if available."""
        if self.history:
            entry = self.history[-1]
            return f"{entry['mood_summary']}. The objectives were: {', '.join(entry['objectives'])}."
        return None

    @function_tool(
        name="complete_checkin",
        description="Call this function ONLY after the user has confirmed their mood/energy and stated their 1-3 objectives for the day. This function saves the log and closes the session.",
    )
    async def complete_checkin(
        self,
        context: RunContext,
        mood_summary: str,
        objectives: List[str],
        advice_given: str
    ):
        """Saves the final check-in details and gives the final recap."""
            
        if not mood_summary or not objectives:
            return "ERROR: Missing required fields. Ask the user for their mood and 1-3 objectives before calling this tool."
            
        # 1. Create the new log entry
        new_entry = {
            "mood_summary": mood_summary,
            "objectives": objectives,
            "agent_reflection": advice_given,
        }
        
        # 2. Persist the data (Uses the MOCK_DATE)
        save_new_entry(new_entry)
        
        # 3. Close the check-in with a brief recap
        print("=" * 50)
        print(f"✅ DAY 3 CHECK-IN COMPLETE! Saved to: {WELLNESS_LOG_FILE} with date {MOCK_DATE}")
        print(json.dumps(new_entry, indent=4))
        print("=" * 50)
        
        objectives_list = ", ".join(objectives)
        final_recap = (
            f"Perfect! So, today's summary is: {mood_summary}. Your goals are: {objectives_list}. "
            "I've saved this entry. Does this sound right? I look forward to our chat tomorrow."
        )
        await context.session.say(final_recap)
        
        return "Log saved and recap delivered. The session is complete."

# --- Existing LiveKit Agent Functions ---

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline 
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

    # Start the session, passing the new WellnessAgent
    await session.start(
        agent=WellnessAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))