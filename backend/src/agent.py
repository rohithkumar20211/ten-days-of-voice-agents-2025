# agent.py
import logging

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
    # function_tool,
    # RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

# Load environment variables from .env.local (same as original)
load_dotenv(".env.local")


class GameMaster(Agent):
    def __init__(self) -> None:
        # System prompt: D&D-style Game Master persona
        super().__init__(
            instructions="""
You are a Dungeons & Dragons–style Game Master running a voice-only adventure.
The entire story takes place in a single universe: a mysterious fantasy world
filled with ancient ruins, magic, forests, dungeons, and strange creatures.

Your role:
- Describe scenes vividly (short, 2–4 sentences).
- Respond only as the Game Master (never break character).
- Continue the story based on the player's spoken choices.
- Maintain continuity using the chat history (remember NPCs, places, decisions).
- After every response, always end with: "What do you do?"

Rules:
1. You control the world, NPCs, environment, and consequences.
2. The player controls only their character and their actions.
3. Keep scenes short and interactive.
4. Ask the player for their next action every turn.
5. If the player is stuck, offer helpful hints or two possible actions.
6. Drive the story toward small narrative arcs (discover a cave, escape danger, find treasure).

Start the game immediately by introducing the player waking up in a mysterious forest.
End the message with: "What do you do?"
"""
        )

    # (Optional) Add any utility function tools here using @function_tool
    # Example weather / dice / inventory tools could be added later.


def prewarm(proc: JobProcess):
    # load voice activity detection model as before
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup - keep original context fields
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Build the voice pipeline session (STT / LLM / TTS / turn detection / vad)
    session = AgentSession(
        # Speech-to-text
        stt=deepgram.STT(model="nova-3"),
        # LLM backend
        llm=google.LLM(
            model="gemini-2.5-flash",
        ),
        # Text-to-speech (Murf). Keep tokenizer/pacing if desired.
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        # Turn detection + VAD
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow preemptive generation while user is speaking
        preemptive_generation=True,
    )

    # Metrics collection - unchanged
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start the session (warm up models) and join the room
    await session.start(
        agent=GameMaster(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Connect to the user (join the voice room)
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
