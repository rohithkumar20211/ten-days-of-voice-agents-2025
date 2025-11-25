import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    RunContext,
    ToolError,
    WorkerOptions,
    cli,
    function_tool,
    metrics,
    tokenize,
)
from livekit.plugins import silero, murf, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

LEARNING_MODES = ("learn", "quiz", "teach_back")
VOICE_PERSONAS = {
    "learn": {
        "voice": "en-US-matthew",
        "style": "Conversation",
        "display": "Matthew",
        "tone": "calm, encouraging explanations",
    },
    "quiz": {
        "voice": "en-US-alicia",
        "style": "Conversation",
        "display": "Alicia",
        "tone": "energetic quiz master",
    },
    "teach_back": {
        "voice": "en-US-ken",
        "style": "Conversation",
        "display": "Ken",
        "tone": "supportive coach who listens closely",
    },
}


@dataclass
class TutorConcept:
    """Structured representation of one concept."""

    id: str
    title: str
    summary: str
    sample_question: str
    teach_back_prompt: str


@dataclass
class ConceptMastery:
    """Simple counters that let the tutor track progress."""

    times_learned: int = 0
    times_quizzed: int = 0
    times_taught_back: int = 0
    last_score: Optional[int] = None
    last_feedback: Optional[str] = None


@dataclass
class TutorSessionState:
    """Conversation-specific session state."""

    current_mode: Optional[str] = None
    current_concept_id: Optional[str] = None
    mastery: Dict[str, ConceptMastery] = field(default_factory=dict)

    def ensure_mastery(self, concept_id: str) -> ConceptMastery:
        if concept_id not in self.mastery:
            self.mastery[concept_id] = ConceptMastery()
        return self.mastery[concept_id]


class TutorContentLibrary:
    """Loads and serves concept content from JSON."""

    def __init__(self, concepts: List[TutorConcept]):
        if not concepts:
            raise ValueError("TutorContentLibrary requires at least one concept.")
        self._concepts: Dict[str, TutorConcept] = {c.id: c for c in concepts}
        self._order: List[str] = [c.id for c in concepts]

    @classmethod
    def from_path(cls, content_path: Path) -> "TutorContentLibrary":
        if not content_path.exists():
            raise FileNotFoundError(f"Tutor content file not found: {content_path}")
        with open(content_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        concepts = [TutorConcept(**item) for item in raw]
        return cls(concepts)

    @classmethod
    def from_env(cls) -> "TutorContentLibrary":
        default_path = Path(__file__).resolve().parents[1] / "shared-data" / "day4_tutor_content.json"
        configured = os.getenv("DAY4_TUTOR_CONTENT_PATH")
        path = Path(configured) if configured else default_path
        return cls.from_path(path)

    def list_concepts(self) -> List[TutorConcept]:
        return [self._concepts[cid] for cid in self._order]

    def get(self, concept_id: Optional[str]) -> TutorConcept:
        target_id = concept_id or self._order[0]
        if target_id not in self._concepts:
            raise KeyError(f"Unknown concept id: {target_id}")
        return self._concepts[target_id]

    def next_concept_id(self, current_id: Optional[str]) -> str:
        if current_id is None:
            return self._order[0]
        try:
            idx = self._order.index(current_id)
        except ValueError:
            return self._order[0]
        return self._order[(idx + 1) % len(self._order)]


@dataclass
class Userdata:
    """Holds both the session state and the content library."""

    state: TutorSessionState
    content: TutorContentLibrary


class TeachTheTutorAgent(Agent):
    """Active recall coach with mode-specific personas."""

    def __init__(self, *, userdata: Userdata) -> None:
        instructions = """You are Teach-the-Tutor, an active recall coach powered by Murf Falcon's fast AI voices.

Your mission: Help users master coding concepts through three learning modes.

WORKFLOW:
1. GREETING (first interaction only):
   - Warmly greet the user
   - Explain the three learning modes:
     * LEARN mode (Matthew's voice) - I explain concepts to you
     * QUIZ mode (Alicia's voice) - She tests your knowledge with questions
     * TEACH BACK mode (Ken's voice) - You explain concepts back to him and he evaluates
   - Ask which mode they'd like to start with
   - Wait for their response

2. MODE SELECTION:
   - When user chooses a mode (learn, quiz, or teach_back), do these steps IN ORDER:
     a) Call set_learning_mode with their chosen mode
     b) Acknowledge the mode switch enthusiastically (e.g., "Great! I've switched to learn mode with Matthew's voice!")
     c) Call list_concepts to get available topics
     d) Tell them the available topics in a friendly way
     e) Ask which topic they want to focus on
   - Wait for them to choose a topic
   - Call set_focus_concept with their choice
   - Now proceed with the mode-specific interaction

3. MODE-SPECIFIC BEHAVIOR:
   
   LEARN MODE (Matthew - calm, encouraging):
   - Call describe_current_concept to get the summary
   - Explain the concept in simple, clear terms
   - Ask if they understand or have questions
   - Offer to quiz them, switch modes, or try another concept
   
   QUIZ MODE (Alicia - energetic, enthusiastic):
   - Call get_quiz_prompt to get the question
   - Ask the question with energy and enthusiasm
   - Listen to their answer
   - Provide immediate, encouraging feedback
   - Offer to continue quizzing, switch modes, or try a new concept
   
   TEACH_BACK MODE (Ken - supportive, patient):
   - Call get_teach_back_prompt to get the instruction
   - Ask the user to explain the concept back to you
   - Listen carefully to their full explanation
   - Evaluate their explanation (score 0-100)
   - Call record_mastery_event with the score and feedback
   - Give encouraging, constructive feedback
   - Tell them what they did well and what to improve

4. MODE SWITCHING:
   - When user asks to switch modes, do these steps:
     a) Call set_learning_mode with the new mode
     b) Acknowledge the switch (e.g., "Switching to quiz mode with Alicia!")
     c) Call list_concepts to show available topics
     d) Ask which topic they want to work on
   - Continue from there

5. PROGRESS TRACKING:
   - If user asks about progress, call get_mastery_snapshot
   - Share encouraging insights

CRITICAL RULES:
- ALWAYS respond after calling tools - never stay silent
- Keep responses conversational (2-4 sentences)
- One concept at a time, one question at a time
- Always use the tools - don't make up content
- Be encouraging and supportive
- Make it fun and engaging!
"""

        super().__init__(instructions=instructions)

    async def on_agent_speech_committed(self, ctx: RunContext[Userdata], message: str) -> None:
        logger.info(f"Agent said: {message}")

    async def on_user_speech_committed(self, ctx: RunContext[Userdata], message: str) -> None:
        logger.info(f"User said: {message}")

    def _require_concept(self, ctx: RunContext[Userdata]) -> TutorConcept:
        state = ctx.userdata.state
        try:
            return ctx.userdata.content.get(state.current_concept_id)
        except KeyError as exc:
            raise ToolError(str(exc)) from exc

    def _apply_voice_persona(self, ctx: RunContext[Userdata], mode: str) -> None:
        """Apply the appropriate Murf Falcon voice for the selected mode."""
        persona = VOICE_PERSONAS[mode]
        tts_engine = ctx.session.tts
        
        if not tts_engine:
            logger.warning("Cannot switch voices: session has no TTS engine configured.")
            return

        update_cb = getattr(tts_engine, "update_options", None)
        if not callable(update_cb):
            logger.warning(
                "TTS engine %s does not support dynamic voice switching. Using initial voice.",
                getattr(tts_engine, "provider", "unknown"),
            )
            return

        try:
            update_cb(voice=persona["voice"], style=persona["style"])
            logger.info("✅ Switched to %s (Murf %s voice) for %s mode", mode, persona["display"], mode)
        except Exception as exc:
            logger.error("❌ Failed to update TTS voice for mode %s: %s", mode, exc)

    @function_tool
    async def list_concepts(self, ctx: RunContext[Userdata]) -> str:
        """List available concepts with their IDs and titles so the learner can choose."""
        concepts = ctx.userdata.content.list_concepts()
        formatted = ", ".join(f"{c.id} ({c.title})" for c in concepts)
        return f"Available concepts: {formatted}. Ask the learner which one they want to focus on."

    @function_tool
    async def set_focus_concept(self, ctx: RunContext[Userdata], concept_id: str) -> str:
        """Set the active concept that the session should focus on."""
        concept = ctx.userdata.content.get(concept_id)
        ctx.userdata.state.current_concept_id = concept.id
        ctx.userdata.state.ensure_mastery(concept.id)
        return f"Concept locked: {concept.title}. You're clear to continue working on {concept.title}."

    @function_tool
    async def describe_current_concept(self, ctx: RunContext[Userdata]) -> str:
        """Return the summary of the current concept for learn mode explanations."""
        concept = self._require_concept(ctx)
        mastery = ctx.userdata.state.ensure_mastery(concept.id)
        mastery.times_learned += 1
        return f"{concept.title}: {concept.summary}"

    @function_tool
    async def get_quiz_prompt(self, ctx: RunContext[Userdata]) -> str:
        """Return a quiz question for the current concept."""
        concept = self._require_concept(ctx)
        mastery = ctx.userdata.state.ensure_mastery(concept.id)
        mastery.times_quizzed += 1
        return f"Quiz question for {concept.title}: {concept.sample_question}"

    @function_tool
    async def get_teach_back_prompt(self, ctx: RunContext[Userdata]) -> str:
        """Return the teach-back instructions for the current concept."""
        concept = self._require_concept(ctx)
        mastery = ctx.userdata.state.ensure_mastery(concept.id)
        mastery.times_taught_back += 1
        return f"Teach this back: {concept.teach_back_prompt}"

    @function_tool
    async def set_learning_mode(self, ctx: RunContext[Userdata], mode: str) -> str:
        """Switch to one of the supported modes: learn, quiz, teach_back."""
        normalized = mode.lower()
        if normalized not in LEARNING_MODES:
            raise ToolError(
                f"Unsupported mode '{mode}'. Choose from: {', '.join(LEARNING_MODES)}."
            )
        ctx.userdata.state.current_mode = normalized
        persona = VOICE_PERSONAS[normalized]
        self._apply_voice_persona(ctx, normalized)
        return (
            f"Switched to {normalized} mode. Murf Falcon voice {persona['display']} is now live "
            f"({persona['tone']}). Let the learner know the new vibe and continue."
        )

    @function_tool
    async def record_mastery_event(
        self,
        ctx: RunContext[Userdata],
        mode: str,
        concept_id: Optional[str] = None,
        score: Optional[int] = None,
        feedback: Optional[str] = None,
    ) -> str:
        """Update mastery stats after finishing an interaction."""
        normalized = mode.lower()
        if normalized not in LEARNING_MODES:
            raise ToolError(f"Mode must be one of {', '.join(LEARNING_MODES)}.")
        target_concept = concept_id or ctx.userdata.state.current_concept_id
        if target_concept is None:
            raise ToolError("Cannot record mastery without an active concept.")
        mastery = ctx.userdata.state.ensure_mastery(target_concept)
        if score is not None:
            mastery.last_score = max(0, min(100, score))
        if feedback:
            mastery.last_feedback = feedback
        return (
            f"Mastery updated for {target_concept} after {normalized} mode. "
            f"Latest score: {mastery.last_score if mastery.last_score is not None else 'n/a'}. "
            f"Feedback noted."
        )

    @function_tool
    async def get_mastery_snapshot(
        self,
        ctx: RunContext[Userdata],
        concept_id: Optional[str] = None,
    ) -> str:
        """Summarize mastery stats for one concept or all of them."""
        if concept_id:
            concepts = [ctx.userdata.content.get(concept_id)]
        else:
            concepts = ctx.userdata.content.list_concepts()
        summaries = []
        for concept in concepts:
            mastery = ctx.userdata.state.mastery.get(concept.id, ConceptMastery())
            summary = (
                f"{concept.title}: learn={mastery.times_learned}, "
                f"quiz={mastery.times_quizzed}, teach_back={mastery.times_taught_back}, "
                f"last_score={mastery.last_score if mastery.last_score is not None else 'n/a'}"
            )
            if mastery.last_feedback:
                summary += f", feedback='{mastery.last_feedback}'"
            summaries.append(summary)
        return " | ".join(summaries)

    @function_tool
    async def advance_to_next_concept(self, ctx: RunContext[Userdata]) -> str:
        """Cycle to the next concept in the content list."""
        next_id = ctx.userdata.content.next_concept_id(ctx.userdata.state.current_concept_id)
        ctx.userdata.state.current_concept_id = next_id
        ctx.userdata.state.ensure_mastery(next_id)
        concept = ctx.userdata.content.get(next_id)
        return f"Advanced to {concept.title}. Let the learner know the new focus."


def prewarm(proc: JobProcess):
    """Prewarm models and load tutor content."""
    proc.userdata["vad"] = silero.VAD.load()
    proc.userdata["tutor_content"] = TutorContentLibrary.from_env()


async def entrypoint(ctx: JobContext):
    """Entry point for Day 4 active recall coach."""
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    content = ctx.proc.userdata.get("tutor_content", TutorContentLibrary.from_env())
    state = TutorSessionState(current_concept_id=content.list_concepts()[0].id)
    userdata = Userdata(state=state, content=content)

    session = AgentSession[Userdata](
        userdata=userdata,
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice=VOICE_PERSONAS["learn"]["voice"],
            style=VOICE_PERSONAS["learn"]["style"],
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

    @session.on("user_speech_committed")
    def _on_user_speech(ev):
        logger.info(f"✅ User speech: {ev.text}")

    @session.on("agent_speech_committed")
    def _on_agent_speech(ev):
        logger.info(f"✅ Agent speech: {ev.text}")

    @session.on("error")
    def _on_error(ev):
        logger.error(f"❌ Session error: {ev}")

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    agent = TeachTheTutorAgent(userdata=userdata)

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()
    logger.info("Day 4 Teach-the-Tutor agent is live and listening.")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))