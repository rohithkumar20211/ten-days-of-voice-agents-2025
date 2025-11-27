import logging
import json
import os
import sqlite3
from typing import Optional, Literal, Dict, Any, Tuple, List

from dotenv import load_dotenv
from livekit.agents import (
    Agent, AgentSession, JobContext, JobProcess, RoomInputOptions,
    WorkerOptions, cli, function_tool, RunContext, llm,
)
from livekit.plugins import murf, google, deepgram, silero, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")

# --- DAY 6: CONFIGURATION & DATABASE SETUP ---
DB_FILE = 'fraud_cases.db'
BANK_NAME = "Falcon Finance"
SECURITY_QUESTION = "What is the 5-digit security identifier associated with the masked card number ending in 4242?"

# List of Case IDs to cycle through for the three scenarios
# Each scenario will use a different ID.
SCENARIO_CASE_IDS = [1, 2, 3] 
CURRENT_RUN_ID = 3 # We will update this global variable before each run

# The base sample fraud case data structure
BASE_FRAUD_CASE_DATA = {
    "user_name": "John Smith", "security_identifier": "12345", 
    "card_ending": "4242", "transaction_amount": 799.99, "merchant_name": "ABC Tech Solutions", 
    "transaction_location": "London, UK", "transaction_time": "approx 11:30 PM IST", 
    "status": "pending_review", "outcome_note": None
}
# ---------------------------------------------

# --- UTILITY FUNCTIONS ---

def init_db():
    """Initializes the SQLite database and ensures all required sample cases exist."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Create the 'cases' table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY, user_name TEXT, security_identifier TEXT, card_ending TEXT, 
            transaction_amount REAL, merchant_name TEXT, transaction_location TEXT, 
            transaction_time TEXT, status TEXT, outcome_note TEXT
        )
    """)
    
    # Insert or reset cases for all SCENARIO_CASE_IDS
    for case_id in SCENARIO_CASE_IDS:
        cursor.execute("SELECT id FROM cases WHERE id = ?", (case_id,))
        if cursor.fetchone() is None:
            # Insert new case with base data and pending status
            case = BASE_FRAUD_CASE_DATA.copy()
            cursor.execute("""
                INSERT INTO cases VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                case_id, case['user_name'], case['security_identifier'], case['card_ending'], 
                case['transaction_amount'], case['merchant_name'], case['transaction_location'], 
                case['transaction_time'], case['status'], case['outcome_note']
            ))
        else:
            # Ensure existing case status is pending_review for a fresh start
            cursor.execute("""
                UPDATE cases SET status = 'pending_review', outcome_note = NULL WHERE id = ?
            """, (case_id,))
            
    conn.commit()
    conn.close()

def load_case(case_id: int) -> Optional[Dict]:
    """Loads a specific fraud case from the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM cases WHERE id = ?", (case_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        cols = ["id", "user_name", "security_identifier", "card_ending", "transaction_amount", 
                "merchant_name", "transaction_location", "transaction_time", "status", "outcome_note"]
        return dict(zip(cols, row))
    return None

def update_case_status(case_id: int, status: str, outcome_note: str):
    """Updates the status and outcome note for a fraud case in the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE cases SET status = ?, outcome_note = ? WHERE id = ?
    """, (status, outcome_note, case_id))
    conn.commit()
    conn.close()
    
    logger.info("=" * 50)
    logger.info(f"✅ DAY 6 STATUS UPDATE: Case ID {case_id} Final Status: {status}")
    logger.info(f"Outcome Note: {outcome_note}")
    logger.info("=" * 50)

# --- AGENT DEFINITION ---

class FraudAgent(Agent):
    def __init__(self, fraud_case: Dict[str, Any]) -> None:
        self.case = fraud_case
        self.bank_name = BANK_NAME
        self.security_answer = self.case['security_identifier']
        self.verification_passed = False 
        
        instructions = (
            f"You are a professional fraud detection representative from {self.bank_name}. "
            "Your call flow MUST be: 1. Introduce yourself and the call's purpose. 2. Verify the customer using the security question provided in the context. 3. If verification passes, read the suspicious transaction details and ask for confirmation (Yes/No). 4. Call the 'mark_transaction_outcome' tool only when the final decision (safe/fraudulent) is known or verification fails. "
            f"Security Question: {SECURITY_QUESTION}. Expected Answer: {self.security_answer}. "
            "Keep language calm and reassuring."
        )

        super().__init__(instructions=instructions)
        self.update_tools([self.mark_transaction_outcome])

    async def on_connected(self) -> None:
        await self.session.say(
            f"Hello, this is the Fraud Detection Department from **{self.bank_name}** calling for **{self.case['user_name']}.** "
            "We detected a suspicious transaction and need to confirm your identity before proceeding. "
            f"To verify, please tell me: **{SECURITY_QUESTION}**"
        )
        
    def _read_transaction_details(self) -> str:
        """Helper to format and read out the suspicious transaction details."""
        currency = "GBP" if "london" in self.case['transaction_location'].lower() else "INR"
        
        return (
            f"Thank you. We detected a transaction on your card ending in {self.case['card_ending']} "
            f"for the amount of **{self.case['transaction_amount']} {currency}** " 
            f"at **{self.case['merchant_name']}** in {self.case['transaction_location']} "
            f"around {self.case['transaction_time']}. "
            "Did you authorize this transaction? Please answer with 'Yes' or 'No'."
        )

    @function_tool
    async def mark_transaction_outcome(
        self, 
        ctx: RunContext, 
        user_response: Literal["yes", "no", "verification_failed"], 
        security_answer_provided: Optional[str] = None
    ) -> str:
        """Tool to handle verification and mark the final outcome."""
        
        global CURRENT_RUN_ID # Use the global variable
        case_id = CURRENT_RUN_ID
        current_status = self.case['status']

        if current_status == "pending_review":
            # --- 1. Verification Logic ---
            if not self.verification_passed:
                if security_answer_provided == self.security_answer:
                    self.verification_passed = True
                    
                    transaction_details = self._read_transaction_details()
                    await ctx.session.say(transaction_details)
                    
                    return "Verification successful. Read transaction details and ask user for confirmation (yes/no). Call this tool again with the final outcome."
                else:
                    # Verification failed
                    update_case_status(case_id, "verification_failed", "Customer failed basic security question.")
                    await ctx.session.say("I'm sorry, I cannot verify your identity with that information. For your security, we cannot proceed with this call. Please call the number on the back of your card.")
                    return "Verification failed. End the call."
            
            # --- 2. Outcome Logic (Verification already passed) ---
            else:
                if user_response == "yes":
                    update_case_status(case_id, "confirmed_safe", "Customer confirmed transaction as legitimate.")
                    await ctx.session.say("Thank you for confirming. We have marked the transaction as legitimate. There is no further action required. Have a safe day.")
                    return "Transaction marked safe. End the call."
                
                elif user_response == "no":
                    update_case_status(case_id, "confirmed_fraud", "Customer denied transaction. Card blocked and dispute initiated (mock).")
                    await ctx.session.say("Understood. We have immediately blocked your card and initiated a dispute for this fraudulent activity. You will receive a follow-up SMS shortly. Thank you for helping us protect your account. Goodbye.")
                    return "Transaction marked fraudulent. End the call."
                
        return "I am waiting for your security answer or a yes/no confirmation. Please provide the missing information."

# --- ENTRYPOINT AND MAIN PROCESS ---

def prewarm(proc: JobProcess):
    """Initializes DB and prewarms VAD."""
    # Initialize the database and ensure the sample cases exist
    init_db() 
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    global CURRENT_RUN_ID # Use global variable to load the correct case
    ctx.log_context_fields = {"room": ctx.room.name}
    
    # Load the specific fraud case based on the global run ID
    fraud_case = load_case(CURRENT_RUN_ID)

    if not fraud_case:
        logger.error(f"Could not load fraud case ID {CURRENT_RUN_ID}. Exiting job.")
        return

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(voice="en-US-matthew", style="Conversation"),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # Initialize the agent with the loaded fraud data
    agent = FraudAgent(fraud_case=fraud_case)

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))