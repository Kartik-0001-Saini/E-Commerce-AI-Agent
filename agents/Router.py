import os
from typing import Any, Dict, List
from langchain_groq import ChatGroq

# Use relative imports because these agents are in the same 'agents' package
from .Customer import CustomerAgent
from .DB import DBAgent
from .Buyer import BuyerAgent
from .Recommendation import RecommendationAgent

class AgentRouter:
    def __init__(self, llm_customer, llm_db, llm_reco):
        # Initialize all agents and hold them for routing
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
        
        return recipient_agent.handle_request(
            sender=sender,
            query=query,
            conversation_history=conversation_history,
            intent=intent
        )
