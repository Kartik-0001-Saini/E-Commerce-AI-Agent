import os
import sqlite3
import uuid
from datetime import datetime
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Import agents
from agents.Buyer import BuyerAgent
from agents.Customer import CustomerAgent
from agents.DB import DBAgent
from agents.Recommendation import RecommendationAgent

# -------------------------------
# Environment & Database Setup
# -------------------------------
load_dotenv()
DB_PATH = "mydb3.db"

# -------------------------------
# Common Utilities (Re-integrated)
# -------------------------------
class CommonUtils:
    @staticmethod
    def clean_sql(sql: str) -> str:
        sql = sql.strip()
        if sql.lower().startswith("```sql"):
            sql = sql[6:]
        if sql.endswith("```"):
            sql = sql[:-3]
        return sql.strip()

    @staticmethod
    def log_conversation(session_id, role, content):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM conversations")
        next_id = cursor.fetchone()[0]
        cursor.execute("""
            INSERT INTO conversations (id, session_id, role, content, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (next_id, session_id, role, content, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        conn.close()

    @staticmethod
    def get_new_session_id() -> str:
        return str(uuid.uuid4())

# -------------------------------
# LLM Setup
# -------------------------------
llm_db = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.getenv("key_sal_groq"))
llm_customer = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.getenv("key_jay_groq"))
llm_reco = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.getenv("key_veer_groq"))

# -------------------------------
# Agent Router (Orchestrator)
# -------------------------------
class AgentRouter:
    def __init__(self):
        # Initialize all agents
        self.customer_agent = CustomerAgent(llm_customer, self)
        self.db_agent = DBAgent(llm_db)
        self.buyer_agent = BuyerAgent(llm_customer)
        self.reco_agent = RecommendationAgent(llm_reco)
        
        self.agents = {
            "CustomerAgent": self.customer_agent,
            "DBAgent": self.db_agent,
            "BuyerAgent": self.buyer_agent,
            "RecommendationAgent": self.reco_agent
        }
        print("[AgentRouter]: Initialized with all agents.")

    def get_agent(self, name: str) -> Any:
        return self.agents.get(name)

    def route_message(self, sender: str, recipient: str, query: str, conversation_history: List[Dict], intent: str = "general_inquiry") -> Dict:
        recipient_agent = self.get_agent(recipient)
        if not recipient_agent:
            return {"content": f"Error: Agent '{recipient}' not found.", "next_agent": "CustomerAgent"}
        
        print(f"[AgentRouter]: Routing message from {sender} to {recipient}. Query: '{query}' Intent: '{intent}'")
        
        # In the modular structure, all agents have a `handle_request` method except CustomerAgent, which has `run`
        if recipient == "CustomerAgent":
             return recipient_agent.run(conversation_history)
        else:
            return recipient_agent.handle_request(
                sender=sender,
                query=query,
                conversation_history=conversation_history,
                intent=intent
            )

# -------------------------------
# Pipeline Execution
# -------------------------------
agent_router = AgentRouter()

def run_pipeline(session_id: str, user_input: str, conversation_history: list, last_candidates=None):
    global agent_router

    if last_candidates:
        agent_router.reco_agent.last_candidates = last_candidates

    active_agent_name = conversation_history[-1].get("next_agent", "CustomerAgent")
    active_agent = agent_router.get_agent(active_agent_name)
    
    current_turn_history = conversation_history + [{"role": "user", "content": user_input, "session_id": session_id}]

    if active_agent_name == "CustomerAgent":
        response_data = active_agent.run(current_turn_history)
    else:
        response_data = active_agent.handle_request(
            sender="User",
            query=user_input,
            conversation_history=current_turn_history,
            intent="general_inquiry"
        )

    final_answer = response_data.get("content", "Sorry, I encountered an issue.")
    next_agent = response_data.get("next_agent", "CustomerAgent")
    
    final_response = {
        "final_answer": final_answer,
        "next_agent": next_agent,
        "last_candidates": agent_router.reco_agent.last_candidates
    }
    
    return final_response