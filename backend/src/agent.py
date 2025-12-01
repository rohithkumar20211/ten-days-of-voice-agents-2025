# agent.py  (Day 10 - corrected: _unused -> unused)
import logging
import json
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
    RunContext,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")


# -----------------------------
# Helper: default improv scenarios
# -----------------------------
DEFAULT_SCENARIOS = [
    "You are a time-travelling tour guide explaining modern smartphones to a visitor from the 1800s. Show them how to 'take a picture' with dramatic flair.",
    "You are a barista who must tell a customer their latte is actually a portal to another dimension — keep it professional but curious.",
    "You are a customer trying to return an obviously cursed object to a very skeptical shop owner. Convince them it's a simple mistake.",
    "You are a detective trying to interrogate a suspect who only answers in song lyrics. Keep the interrogation focused.",
    "You are a waiter apologizing because the customer's order has literally run away. Explain what happened and try to offer a replacement."
]


class ImprovHost(Agent):
    def __init__(self, max_rounds: int = 3) -> None:
        # System prompt defines the improv-host persona and rules
        super().__init__(
            instructions=f"""
You are the host of a high-energy TV improv show called "Improv Battle".
Your persona:
- Energetic, witty, playful, sometimes teasing, always respectful.
- Give short scene setups, invite the player to perform, then react.
- Use varied reactions (supportive / neutral / mild critique) across rounds.

STRUCTURE & RULES:
1) On session start: introduce the show, explain rules briefly, ask player's name (or use provided name).
2) There will be up to {max_rounds} improv rounds.
   - For each round:
     a) Call get_next_scenario() (a tool) to fetch the scenario JSON.
     b) Announce the scenario clearly, then prompt the player: "Start improv — when you're done say 'End scene'."
     c) Listen to the player's performance. When the player ends (they say 'End scene' or the LLM detects an end cue), produce a reaction line that:
        - references something concrete from the player's performance when possible,
        - randomly varies tone (praise / constructive critique / playful tease),
        - ends with a short transition to the next round or closing.
     d) Call save_reaction(reaction_json) to persist the reaction + player snippet.
3) If the player says "stop game", "end show", or similar, confirm and end gracefully.
4) At the end (after max rounds), provide a short closing summary:
   - Describe the player's improv strengths (character, absurdity, emotional range, timing).
   - Mention 1–2 specific moments that stood out (use stored state).
   - Thank the player and close the show.

IMPORTANT: Use the provided tools (get_next_scenario and save_reaction) to manage the backend state. Do not invent the state format — follow the tool responses exactly.
"""
        )
        # session state (per-agent instance)
        self.improv_state = {
            "player_name": None,
            "current_round": 0,
            "max_rounds": max_rounds,
            "rounds": [],  # each entry: {"scenario": str, "player_excerpt": str, "host_reaction": str, "timestamp": str}
            "phase": "intro",  # intro | awaiting_improv | reacting | done
            "started_at": None,
        }

    # -----------------------------
    # TOOL: get_next_scenario
    # -----------------------------
    @function_tool
    async def get_next_scenario(self, ctx: RunContext, unused: str = ""):
        """
        Returns the next scenario JSON and updates improv_state.phase -> 'awaiting_improv'.
        The LLM should call this to get the scenario text to read to the player.
        Response format: {"scenario": "<text>", "round_number": n}
        """
        if self.improv_state["current_round"] >= self.improv_state["max_rounds"]:
            return json.dumps({"error": "no_more_rounds"})

        # pick scenario (cycle through defaults)
        idx = self.improv_state["current_round"] % len(DEFAULT_SCENARIOS)
        scenario = DEFAULT_SCENARIOS[idx]

        # increment current_round and set phase
        self.improv_state["current_round"] += 1
        self.improv_state["phase"] = "awaiting_improv"
        if not self.improv_state["started_at"]:
            self.improv_state["started_at"] = datetime.utcnow().isoformat() + "Z"

        response = {
            "scenario": scenario,
            "round_number": self.improv_state["current_round"],
        }
        return json.dumps(response)

    # -----------------------------
    # TOOL: save_reaction
    # -----------------------------
    @function_tool
    async def save_reaction(self, ctx: RunContext, reaction_json: str):
        """
        Persist the host reaction to the improv_state and return the updated round record.
        reaction_json example:
         {
           "scenario": "...",
           "player_excerpt": "player said ...",
           "host_reaction": "That was funny because ...",
           "outcome": "continue" | "stop"
         }
        """
        try:
            payload = json.loads(reaction_json)
        except Exception as e:
            return json.dumps({"error": "invalid_json", "message": str(e)})

        # build round record
        entry = {
            "scenario": payload.get("scenario"),
            "player_excerpt": payload.get("player_excerpt"),
            "host_reaction": payload.get("host_reaction"),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        self.improv_state["rounds"].append(entry)
        # after saving reaction, set phase depending on outcome
        outcome = payload.get("outcome", "continue")
        if outcome == "stop" or self.improv_state["current_round"] >= self.improv_state["max_rounds"]:
            self.improv_state["phase"] = "done"
        else:
            self.improv_state["phase"] = "reacting"

        return json.dumps({"saved": True, "round_index": len(self.improv_state["rounds"]) - 1})

    # -----------------------------
    # Optional helper: retrieve state (not exposed as a tool by default)
    # -----------------------------
    def get_state(self):
        return self.improv_state


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # logging fields for session
    ctx.log_context_fields = {"room": ctx.room.name}
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Promo",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
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
        logger.info(f"Usage: {usage_collector.get_summary()}")

    ctx.add_shutdown_callback(log_usage)

    # start the session with our ImprovHost agent
    await session.start(
        agent=ImprovHost(max_rounds=3),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))

    
