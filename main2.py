# import os
# from typing import Any, Dict, List
# from dotenv import load_dotenv
# from langchain_groq import ChatGroq

# # Import agents and utils

# from agents.Buyer import BuyerAgent
# from agents.Customer import CustomerAgent
# from agents.DB import DBAgent
# from agents.Recommendation import RecommendationAgent

# # -------------------------------
# # Environment & LLM Setup
# # -------------------------------
# load_dotenv()
# llm_db = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.getenv("key_sal_groq"))
# llm_customer = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.getenv("key_jay_groq"))
# llm_reco = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.getenv("key_veer_groq"))

# # -------------------------------
# # Agent Router (Orchestrator)
# # -------------------------------
# class AgentRouter:
#     def __init__(self):
#         # Initialize all agents and hold them for routing
#         self.customer_agent = CustomerAgent(llm_customer, self)
#         self.db_agent = DBAgent(llm_db)
#         self.buyer_agent = BuyerAgent(llm_customer)
#         self.reco_agent = RecommendationAgent(llm_reco)
        
#         self.agents = {
#             "CustomerAgent": self.customer_agent,
#             "DBAgent": self.db_agent,
#             "BuyerAgent": self.buyer_agent,
#             "RecommendationAgent": self.reco_agent
#         }
#         print("[AgentRouter]: Initialized with all agents.")

#     def get_agent(self, name: str) -> Any:
#         return self.agents.get(name)

#     def route_message(self, sender: str, recipient: str, query: str, conversation_history: List[Dict], intent: str = "general_inquiry") -> Dict:
#         recipient_agent = self.get_agent(recipient)
#         if not recipient_agent:
#             return {"content": f"Error: Agent '{recipient}' not found.", "next_agent": "CustomerAgent"}
        
#         print(f"[AgentRouter]: Routing message from {sender} to {recipient}. Query: '{query}' Intent: '{intent}'")
        
#         return recipient_agent.handle_request(
#             sender=sender,
#             query=query,
#             conversation_history=conversation_history,
#             intent=intent
#         )

# # -------------------------------
# # Pipeline Execution
# # -------------------------------
# agent_router = AgentRouter()

# def run_pipeline(session_id: str, user_input: str, conversation_history: list, last_candidates=None):
#     global agent_router

#     # Restore last candidates to the recommendation agent if they exist
#     if last_candidates:
#         agent_router.reco_agent.last_candidates = last_candidates

#     active_agent_name = conversation_history[-1].get("next_agent", "CustomerAgent")
#     active_agent = agent_router.get_agent(active_agent_name)
    
#     # Add current user input to history for the agent to process
#     current_turn_history = conversation_history + [{"role": "user", "content": user_input, "session_id": session_id}]

#     if active_agent_name == "CustomerAgent":
#         response_data = active_agent.run(current_turn_history)
#     else:
#         response_data = active_agent.handle_request(
#             sender="User",
#             query=user_input,
#             conversation_history=current_turn_history,
#             intent="general_inquiry" # Intent is determined by CustomerAgent, so this is a default
#         )

#     final_answer = response_data.get("content", "Sorry, I encountered an issue.")
#     next_agent = response_data.get("next_agent", "CustomerAgent")
    
#     # Prepare the final response dictionary
#     final_response = {
#         "final_answer": final_answer,
#         "next_agent": next_agent,
#         "last_candidates": agent_router.reco_agent.last_candidates # Always return the latest candidates
#     }
    
#     return final_response
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
load_dotenv()
llm_db = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.getenv("key_sal_groq"))
llm_customer = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.getenv("key_jay_groq"))
llm_reco = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.getenv("key_veer_groq"))
from pipeline.AGentic_pipeline import run_pipeline

if __name__ == "__main__":
    run_pipeline()