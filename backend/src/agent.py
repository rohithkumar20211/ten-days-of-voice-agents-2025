import logging
import json
import os
from datetime import datetime
from typing import Optional, Literal, List, Dict, Any

# Ensure .env.local is loaded first
from dotenv import load_dotenv
load_dotenv(".env.local")

# Fix: Ensure all necessary classes are imported explicitly
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess, 
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
    llm,
)
from livekit.plugins import murf, google, deepgram, silero, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")


# --- DAY 5: CONFIGURATION ---
# CRITICAL: Ensure this file exists in backend/src/sdr-content/
FAQ_FILE = "src/sdr-content/ola_faq.json"
LEAD_OUTPUT_FILE = "leads_log.json"

# Define the structured Lead Data fields
INITIAL_LEAD_STATE = {
    "name": None,
    "company": None,
    "email": None,
    "role": None,
    "use_case": None,
    "team_size": None,
    "timeline": None,
    "faq_answered": 0,
}

# --- UTILITY FUNCTIONS ---

def load_faq_content() -> Dict[str, Any]:
    """Loads FAQ content for RAG."""
    # Ensure the directory exists when loading (best practice)
    if not os.path.exists(os.path.dirname(FAQ_FILE)):
        os.makedirs(os.path.dirname(FAQ_FILE))
        
    try:
        with open(FAQ_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load FAQ content: {e}")
        return {}

def simple_faq_search(faq_content: List[Dict], user_query: str) -> Optional[str]:
    """Performs a simple keyword search over FAQ keywords."""
    query = user_query.lower()
    for entry in faq_content:
        # Check if any keyword in the FAQ entry is present in the user's query
        if any(keyword in query for keyword in entry.get("keywords", [])):
            return entry["answer"]
    return None

# --- AGENT DEFINITION ---

class SdrAgent(Agent):
    def __init__(self, faq_data: Dict[str, Any]) -> None:
        self.faq_data = faq_data
        self.lead_state = INITIAL_LEAD_STATE.copy()
        
        company = faq_data.get("company_name", "Our Company")
        pitch = faq_data.get("sdr_pitch", "a cutting-edge product.")
        
        # Set up the SDR persona (Prompt Designing)
        sdr_instructions = (
            f"You are a friendly Sales Development Representative (SDR) for {company}'s platform. "
            "Your main goal is to understand the visitor's needs and collect ALL lead information (Name, Company, Email, Role, Use Case, Team Size, and Timeline). "
            "**CRITICAL RULE: Call 'update_lead_state' immediately every time the user provides a detail (Name, Email, Role, Use Case, etc.) to store it.** "
            f"Start with a warm greeting and ask what they are working on. Use the 'answer_faq' tool to address questions about the product, pricing, or audience. "
            "If the user says 'That's all' or 'I'm done', call the 'save_lead_summary' tool immediately. "
            f"Company pitch: {pitch}"
        )

        super().__init__(instructions=sdr_instructions)
        # Expose all three tools
        self.update_tools([self.answer_faq, self.save_lead_summary, self.update_lead_state]) 

    async def on_connected(self) -> None:
        await self.session.say(
            f"Hello! Thank you for visiting {self.faq_data['company_name']}. I'm your dedicated SDR. "
            "What brought you here today, and what project are you currently working on?"
        )

    @function_tool
    async def answer_faq(self, ctx: RunContext, query: str) -> str:
        """Looks up the user's question in the FAQ content and returns the answer."""
        
        answer = simple_faq_search(self.faq_data.get("faqs", []), query)
        
        if answer:
            self.lead_state['faq_answered'] = self.lead_state.get('faq_answered', 0) + 1
            
            # CRITICAL: After answering, the LLM must call update_lead_state next to prompt for details.
            return f"Yes, I can tell you that: {answer}. Now, based on what you told me, please call the 'update_lead_state' tool next to gather your details."
        
        return "I apologize, I don't have that specific information in my current knowledge base. Is there anything else I can help you with regarding our product?"

    @function_tool
    async def update_lead_state(
        self,
        ctx: RunContext,
        name: Optional[str] = None,
        company: Optional[str] = None,
        email: Optional[str] = None,
        role: Optional[str] = None,
        use_case: Optional[str] = None,
        team_size: Optional[str] = None,
        timeline: Optional[str] = None,
    ) -> str:
        """Called by the LLM every time the user provides any piece of lead information (name, email, role, etc.)."""
        
        # 1. Update the state with extracted arguments
        updates = {k: v for k, v in locals().items() if v is not None and k in self.lead_state}
        self.lead_state.update(updates)
        
        # 2. Check for missing fields to prompt the LLM for the next step
        missing_fields = [k.replace('_', ' ') for k, v in self.lead_state.items() if v is None and k != 'faq_answered']
        
        if missing_fields:
            # Tell the LLM what is still missing so it can ask for the next piece of data
            return f"Lead state updated. The following is still missing: {', '.join(missing_fields)}. Please politely ask the user for the next missing piece of information."
        else:
            return "Lead state is complete. Proceed with the final summary or ask if the user has other questions."

    @function_tool
    async def save_lead_summary(self, ctx: RunContext) -> str:
        """Gives a final verbal summary and saves the complete lead data to a JSON file."""
        
        # 1. Verbal Summary (using the now-updated self.lead_state)
        name = self.lead_state.get("name", "a valuable prospect")
        use_case = self.lead_state.get("use_case", "unspecified needs")
        timeline = self.lead_state.get("timeline", "later")
        
        final_summary = (
            f"Thank you, {name}. I've captured your details. Just to quickly recap: "
            f"You are interested in using our platform for **{use_case}**, and your rough timeline is **{timeline}**. "
            "I'm saving this summary now for our sales team."
        )
        await ctx.session.say(final_summary)

        # 2. Save to JSON
        try:
            if os.path.exists(LEAD_OUTPUT_FILE):
                with open(LEAD_OUTPUT_FILE, 'r+') as f:
                    data = json.load(f)
                    if not isinstance(data, list):
                        data = []
                    data.append(self.lead_state)
                    f.seek(0)
                    json.dump(data, f, indent=4)
            else:
                with open(LEAD_OUTPUT_FILE, 'w') as f:
                    json.dump([self.lead_state], f, indent=4)
            
            # Print confirmation to the terminal (for video recording proof)
            logger.info("=" * 50)
            logger.info(f"✅ DAY 5 PRIMARY GOAL COMPLETE! Lead Saved to: {LEAD_OUTPUT_FILE}")
            logger.info(json.dumps(self.lead_state, indent=4))
            logger.info("=" * 50)

            return "Lead saved successfully. Thank the user and end the call warmly."
        except Exception as e:
            return f"An error occurred while saving the lead data: {e}. Please try again."

# --- ENTRYPOINT AND MAIN PROCESS ---

def prewarm(proc: JobProcess):
    """Prewarm models and load SDR content."""
    proc.userdata["vad"] = silero.VAD.load()
    # Load FAQ content into worker process memory
    proc.userdata["faq_data"] = load_faq_content()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    
    faq_data = ctx.proc.userdata.get("faq_data")

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(
            model="gemini-2.5-flash",
            # We rely on the CRITICAL RULE and update_lead_state tool, 
            # as extract_context_fields is not supported in this SDK version.
        ),
        tts=murf.TTS(voice="en-US-matthew", style="Conversation"),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # Initialize the agent with the loaded FAQ data
    agent = SdrAgent(faq_data=faq_data)

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))