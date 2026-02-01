from __future__ import annotations
import json
import re
from typing import Dict, List, TYPE_CHECKING

# from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate

if TYPE_CHECKING:
    from .Router import AgentRouter

class CustomerAgent:
    def __init__(self, llm, router: AgentRouter):
        self.llm = llm
        self.router = router
        self.system_prompt = ChatPromptTemplate.from_template("""
        You are the lead e-commerce Customer Support Agent. Your job is to understand the user's request and delegate it to the correct specialist agent.

        **Your Specialist Team:**
        1.  **DBAgent**: For questions about existing data like order status, product stock, price, etc.
        2.  **BuyerAgent**: When the user wants to buy or order a new product.
        3.  **RecommendationAgent**: For comparing products or asking for recommendations.

        **Your Process:**
        1.  Analyze the user's message.
        2.  Choose the single best agent.
        3.  Respond ONLY with a valid JSON object. Do NOT include any extra text or markdown.
        4.  Always include the `intent`.

        **JSON Response Format:**
        {{
            "action": "delegate",
            "recipient": "AGENT_NAME",
            "query": "A clear query for the specialist agent.",
            "intent": "one of [order_status, returns, product_details, general_inquiry, purchase_orders, order_purchases]"
        }}

        **Conversation History:**
        {history}
        **User's Latest Message:**
        {input}
        """)

    def _extract_json_from_response(self, text: str) -> str:
        match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            return match.group(1)
        try:
            start_index = text.index('{')
            end_index = text.rindex('}') + 1
            return text[start_index:end_index]
        except ValueError:
            return ""

    def run(self, conversation_history: List[Dict]) -> Dict:
        print("[CustomerAgent]: Analyzing user request...")
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])
        prompt = self.system_prompt.format(history=history_str, input=conversation_history[-1]['content'])

        try:
            raw_response = self.llm.invoke(prompt).content.strip()
            print(f"[CustomerAgent]: LLM Raw Response: {raw_response}")
            clean_json_str = self._extract_json_from_response(raw_response)
            if not clean_json_str:
                raise ValueError("No JSON object found in the response.")
            decision = json.loads(clean_json_str)
            print(f"[CustomerAgent]: Successfully Parsed Decision: {decision}")
        except Exception as e:
            print(f"[CustomerAgent]: Error parsing LLM decision. Defaulting to direct response. Error: {e}")
            return {"content": "I'm sorry, I'm having a little trouble understanding. Could you please rephrase?", "next_agent": "CustomerAgent"}

        if decision.get("action") == "delegate":
            recipient = decision.get("recipient")
            if recipient == "CustomerAgent":
                return {"content": decision.get("query"), "next_agent": "CustomerAgent"}
            return self.router.route_message(
                sender="CustomerAgent",
                recipient=recipient,
                query=decision.get("query"),
                conversation_history=conversation_history,
                intent=decision.get("intent", "general_inquiry")
            )
        else:
            return {"content": "I'm not sure how to handle that request. Please try again.", "next_agent": "CustomerAgent"}