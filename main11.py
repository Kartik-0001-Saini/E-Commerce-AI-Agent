# -------------------------------
# Imports
# -------------------------------
import re
import sqlite3
from datetime import datetime
from typing import Any, Dict, List
import uuid
import json
import os

# LangChain Core
from langchain_core.tools import tool

# LangChain Community
from langchain_community.utilities import SQLDatabase
from langchain_community.tools import DuckDuckGoSearchRun

# LangChain Groq
from langchain_groq import ChatGroq

# LangChain Prompts
from langchain.prompts import ChatPromptTemplate

from dotenv import load_dotenv

# -------------------------------
# Environment & Database Setup
# -------------------------------
load_dotenv()
DB_PATH = "mydb3.db"
db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

# -------------------------------
# LLMs
# -------------------------------
# For consistency and better management, using a single LLM instance is often better.
# However, we will keep them separate as per the original structure for now.
llm_db = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.getenv("key_sal_groq"))
llm_customer = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.getenv("key_jay_groq"))
llm_reco = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.getenv("key_veer_groq"))

# -------------------------------
# Common Utilities
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
        # This function can be expanded for more robust logging
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
# Agent Router (The new orchestrator)
# -------------------------------
class AgentRouter:
    def __init__(self):
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

        # Pass intent into agent request
        return recipient_agent.handle_request(
            sender=sender,
            query=query,
            conversation_history=conversation_history,
            intent=intent
        )
# -------------------------------
# Buyer Agent (Handles multi-step purchasing)
# -------------------------------
class BuyerAgent:
    def __init__(self, llm):
        self.llm = llm
        self.purchase_sessions = {} # In-memory state for ongoing purchases

    def handle_request(self, sender: str, query: str, conversation_history: List[Dict],intent:str=None) -> Dict:
        session_id = conversation_history[-1].get("session_id")
        user_text = conversation_history[-1].get("content")

        if session_id not in self.purchase_sessions:
            print(f"[BuyerAgent]: Initializing new purchase session for {session_id}")
            self.purchase_sessions[session_id] = self._init_purchase_state(session_id)
            self.purchase_sessions[session_id]["question"] = query # Use initial query from CustomerAgent
        else:
            self.purchase_sessions[session_id]["question"] = user_text # Use latest user message

        pstate = self.purchase_sessions[session_id]
        reply = self._process_user_message(pstate, user_text)

        if pstate.get("purchase_stage") in ("completed", "discarded"):
            print(f"[BuyerAgent]: Purchase session {session_id} ended. Returning control to CustomerAgent.")
            del self.purchase_sessions[session_id]
            next_agent = "CustomerAgent"
        else:
            print(f"[BuyerAgent]: Purchase session {session_id} ongoing. BuyerAgent remains active.")
            next_agent = "BuyerAgent"

        return {"content": reply, "next_agent": next_agent}
    def compute_and_build_summary(self, state: dict) -> str:
        """Build an order summary and prepare for confirmation."""
        unit_price = state["selected_product"]["price"]
        total_units = state["total_units"]
        base_cost = unit_price * total_units
        transportation_cost = base_cost*0.12  # fixed example, can make dynamic
        total_cost = base_cost + transportation_cost

        summary = {
            "buyer_id": str(uuid.uuid4())[:8],  # short ID
            "name": state["buyer_name"],
            "location": state["buyer_location"],
            "contact": state["buyer_contact"],
            "product_id": state["selected_product"]["product_id"],
            "product_name": state["selected_product"]["name"],
            "unit_price": unit_price,
            "total_units": total_units,
            "transportation_cost": transportation_cost,
            "base_cost": base_cost,
            "total_cost": total_cost,
            "status": "Awaiting Confirmation",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        state["summary"] = summary
        state["purchase_stage"] = "awaiting_confirmation"

        lines = [
            "📝 Order Summary:",
            f"- Buyer: {summary['name']} ({summary['contact']}, {summary['location']})",
            f"- Product: {summary['product_name']} (ID: {summary['product_id']})",
            f"- Units: {summary['total_units']}",
            f"- Unit Price: ₹{int(summary['unit_price']):,}",
            f"- Base Cost: ₹{int(summary['base_cost']):,}",
            f"- Transportation Cost: ₹{int(summary['transportation_cost']):,}",
            f"- Total Cost: ₹{int(summary['total_cost']):,}",
            "",
            "Do you want to confirm this order? (yes/no)"
        ]
        return "\n".join(lines)

    def save_buyer_order(self, buyer_record: dict):
        """Save confirmed buyer order into the buyer table."""
        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO buyer (
                    buyer_id, name, location, contact, product_id, product_name,
                    unit_price, total_units, transportation_cost, base_cost, total_cost,
                    status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                buyer_record["buyer_id"],
                buyer_record["name"],
                buyer_record["location"],
                buyer_record["contact"],
                buyer_record["product_id"],
                buyer_record["product_name"],
                buyer_record["unit_price"],
                buyer_record["total_units"],
                buyer_record["transportation_cost"],
                buyer_record["base_cost"],
                buyer_record["total_cost"],
                buyer_record.get("status", "Order Placed"),
                buyer_record.get("created_at") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))
            conn.commit()
            conn.close()
            print(f"[BuyerAgent]: Order saved for Buyer ID {buyer_record['buyer_id']}")
        except Exception as e:
            print(f"[BuyerAgent]: Error saving order → {e}")

    def _init_purchase_state(self, session_id: str) -> Dict:
        return {"session_id": session_id, "purchase_stage": None, "purchase_candidates": None, "selected_product": None, "total_units": None, "buyer_name": None, "buyer_location": None, "buyer_contact": None, "summary": None}
    
    def _get_product_keyword(self, text: str) -> str:
        try:
            prompt = f"""Extract the main product keyword from: "{text}". Return only one word or phrase."""
            return self.llm.invoke(prompt).content.strip().lower()
        except Exception:
            return text.split()[-1]

    def _present_options_and_update(self, state: Dict):
        user_text = state.get("question", "")
        keyword = self._get_product_keyword(user_text)
        # We instantiate RecommendationAgent here just to use its fetch_products utility.
        candidates = RecommendationAgent(llm_reco).fetch_products(search=keyword, limit=3)
        if not candidates:
            return f"Sorry, no matching products found for '{keyword}'."
        state["purchase_stage"] = "awaiting_selection"
        state["purchase_candidates"] = candidates
        lines = ["We have these products:"]
        for i, p in enumerate(candidates, start=1):
            lines.append(f"Option {i}: {p['name']} — ₹{int(p['price']):,} (stock: {p['stock']})")
        lines.append("Which one would you like? (e.g., 'option 2')")
        return "\n".join(lines)
    
    # Other internal methods for purchase flow...
    def _process_user_message(self, state: dict, user_text: str):
        text = user_text.strip().lower()
        stage = state.get("purchase_stage")

        if not stage:
            return self._present_options_and_update(state)
        
        if stage == "awaiting_selection":
            selection = -1
            if "1" in text or "one" in text: selection = 0
            elif "2" in text or "two" in text: selection = 1
            elif "3" in text or "three" in text: selection = 2
            
            if 0 <= selection < len(state["purchase_candidates"]):
                state["selected_product"] = state["purchase_candidates"][selection]
                state["purchase_stage"] = "awaiting_quantity"
                return f"You selected {state['selected_product']['name']}. How many units would you like?"
            else:
                return "Please choose a valid option (e.g., 'option 1')."
        if stage == "awaiting_quantity":
            if text.isdigit():
                state["total_units"] = int(text)
                state["purchase_stage"] = "awaiting_buyer_name"
                return "Great! What is your name?"
            else:
                return "Please enter a valid number."

        if stage == "awaiting_buyer_name":
            state["buyer_name"] = user_text.strip()
            state["purchase_stage"] = "awaiting_buyer_location"
            return f"Thanks {state['buyer_name']}. Which city do you live in?"

        if stage == "awaiting_buyer_location":
            state["buyer_location"] = user_text.strip()
            state["purchase_stage"] = "awaiting_buyer_contact"
            return "Got it. Please provide your contact details."

        if stage == "awaiting_buyer_contact":
            state["buyer_contact"] = user_text.strip()
            return self.compute_and_build_summary(state)

        if stage == "awaiting_confirmation":
            if "yes" in text:
                rec = state["summary"]
                rec["status"] = "Order Placed"
                self.save_buyer_order(rec)
                state["purchase_stage"] = "completed"
                return f"✅ Order placed successfully! Buyer ID: {rec['buyer_id']}"
            elif "no" in text:
                state["purchase_stage"] = "discarded"
                return "❌ Order discarded."
            else:
                return "Please reply 'yes' or 'no'."
        return "Purchase flow is in progress." # Placeholder for brevity

# -------------------------------
# Recommendation Agent (Now a full agent)
# -------------------------------
class RecommendationAgent:
    def __init__(self, llm):
        self.llm = llm
        self.last_candidates = []  # ✅ memory for last shown products
    def run(self, session_id, message, conversation_history=None):
        # Special handling: supplier queries
        if "company" in message.lower() or "supplier" in message.lower():
            if "sell" in message.lower() or "list" in message.lower():
                # Extract product name (simplified, can improve with LLM if needed)
                words = message.split()
                product = words[-1]  # last word assumption, e.g. "headphone"
                return {"final_answer": self.db_agent.get_suppliers_for_product(product),
                        "next_agent": "DBAgent"}
    def handle_request(self, sender: str, query: str, conversation_history: List[Dict], intent: str = None) -> Dict:
        print(f"[RecoAgent]: Received request from {sender}: '{query}'")

        # If query looks like a comparison → jump to comparison
        if "compare" in query.lower() or "vs" in query.lower():
            response = self._generate_comparison(query)
        else:
            # Otherwise fetch some products
            products = self.fetch_products(search=query, limit=5)
            if products:
                self.last_candidates = products  # ✅ store in memory
                lines = ["We have a variety of products available for you to choose from. Here are the options:"]
                for i, p in enumerate(products, start=1):
                    lines.append(
                        f"{i}. {p['name']} (Product ID: {p['product_id']}) - Priced at ₹{int(p['price']):,}, stock: {p['stock']}. {p['description']}"
                    )
                lines.append("If you'd like more information, or want me to compare any of these, just let me know!")
                response = "\n".join(lines)
            else:
                response = "Sorry, I couldn’t find any matching products."

        return {"content": response, "next_agent": "CustomerAgent"}
    def safe_json_extract(text: str):
        """
        Extract valid JSON from messy LLM output.
        """
        try:
            return json.loads(text)
        except:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except:
                    return {}
            return {}
    def fetch_products(self, search: str = "", limit: int = 5):
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("""
            SELECT p.product_id, p.name, p.price, p.stock, p.description,
                   s.name AS supplier_name, s.warehouse_location, s.feature, s.discount_value
            FROM products p
            JOIN suppliers s ON p.supplier_id = s.supplier_id
            WHERE LOWER(p.name) LIKE ?
            LIMIT ?
        """, (f'%{search.lower()}%', limit))
        rows = cur.fetchall()
        conn.close()
        return [
            {
                "product_id": r[0],
                "name": r[1],
                "price": r[2],
                "stock": r[3],
                "description": r[4],
                "supplier_name": r[5],
                "warehouse_location": r[6],
                "feature": r[7],
                "discount_value": r[8],
            }
            for r in rows
        ]

    def _generate_comparison(self, query: str) -> str:
        # ✅ Use last shown products
        if not self.last_candidates:
            return "I don’t have a list of products to compare yet. Please ask me to show some products first."

        extraction_prompt = f"""
        You are a helpful assistant. The user has been shown this list of products (with supplier details):

        {json.dumps(self.last_candidates, indent=2)}

        The user query is:
        "{query}"

        Task:
        - Figure out which of the above products the user wants to compare.
        - The user might refer by number (1, 2, 3...), by name (e.g. "headphones"), by product_id, or by saying "all".
        - Always return a JSON array of the exact products (with all supplier details included).
        """

        try:
            raw = self.llm.invoke(extraction_prompt).content.strip()
            products_to_compare = json.loads(raw)
        except Exception as e:
            return f"⚠️ I had trouble figuring out which products to compare. ({e})"

        if len(products_to_compare) < 2:
            return "I couldn’t find enough products to compare. Please mention at least two."

        comparison_prompt = ChatPromptTemplate.from_template("""
        You are a product comparison expert. Create a clear markdown table comparing these products and give a recommendation.

        Products (JSON list):
        {products}

        User Query: "{query}"

        **Comparison Table and Recommendation:**
        """).format(products=json.dumps(products_to_compare, indent=2), query=query)

        try:
            return self.llm.invoke(comparison_prompt).content.strip()
        except Exception as e:
            return f"Error while generating comparison: {e}"

# -------------------------------
# DB Agent (Handles database queries)
# -------------------------------
class DBAgent:
    def __init__(self, llm):
        self.llm = llm
        self.db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
    def get_suppliers_for_product(self, product_name: str):
        """
        Fetch all supplier details for a given product.
        """
        query = """
        SELECT DISTINCT s.supplier_id, s.name, s.warehouse_location,
                        s.product_name, s.feature, s.discount_value
        FROM suppliers s
        WHERE s.product_name LIKE ?
        """
        rows = self.execute_sql(query, (f"%{product_name}%",))
        if not rows:
            return f"⚠️ No suppliers found for {product_name}."

        details = []
        for r in rows:
            details.append(f"""
🏢 **Supplier**: {r[1]}
📍 **Warehouse Location**: {r[2]}
📦 **Product**: {r[3]}
✨ **Feature**: {r[4]}
💸 **Discount**: {r[5]}%
""")
        return "\n".join(details)
    def handle_request(self, sender: str, query: str, conversation_history: List[Dict], intent: str="general_inquiry") -> Dict:
        print(f"[DBAgent]: Received request from {sender}: '{query}' with intent: {intent}")
        sql_query = self._generate_sql(query, intent)
        
        if "Error:" in sql_query or "unsafe" in sql_query.lower():
            return {"content": sql_query, "next_agent": "CustomerAgent"}

        results = self._execute_sql(sql_query)
        if isinstance(results, list) and results and "product_id" in results[0]:
            try:
                # update reco agent memory
                from main11 import agent_router  # if circular import not an issue
                reco_agent = agent_router.get_agent("RecommendationAgent")
                if reco_agent:
                    reco_agent.last_candidates = results
                    print("[DBAgent]: Updated RecommendationAgent.last_candidates")
            except Exception as e:
                print(f"[DBAgent]: Could not update reco_agent → {e}")
        if "Error:" in results:
            return {"content": results, "next_agent": "CustomerAgent"}

        response = self._format_response(original_query=query, results=results)
        return {"content": response, "next_agent": "CustomerAgent"} # Always returns control

    def _generate_sql(self, question: str,intent) -> str:
        schema = self.db.get_table_info()
        question_for_sql = question
        sql_prompt_intent = ChatPromptTemplate.from_template("""
        You are an SQL expert for an e-commerce support chatbot.

        Database Schema:
        {schema}

        Rules:
        - Only output a single SELECT statement.
        - Do NOT include explanations or formatting.
        - Only use tables/columns from the schema.
        - The intent of the query is: {intent}
        - Always join products with suppliers using supplier_id when needed.
        - Never guess supplier names or product names.
        - Do not fabricate results — only use data from the database.

        Extra rules by intent:
        - For order_status → select: status, tracking_number, created_at
        - For returns → select: status, reason, created_at
        - For product_details → select:product_id, name, price, stock, description
        - For general_inquiry (about suppliers) → select: supplier_id, name, product_name, warehouse_location, feature, discount_value
        - For purchase_orders → select: purchase_id, product_id, supplier_id, supplier_name, delivery_time, product_name, product_price
        - For order_purchases → select: id, order_id, purchase_id
        User asked: {question}
        SQL Query:
        """)

        sql_raw = self.llm.invoke(sql_prompt_intent.format(
            schema=schema,
            intent=intent,
            question=question_for_sql
        )).content.strip()

        return CommonUtils.clean_sql(sql_raw)


    def _execute_sql(self, sql: str) -> str:
        if any(word in sql.upper() for word in ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER"]):
            return "Error: Unsafe SQL detected. Only SELECT queries are allowed."
        try:
            conn = sqlite3.connect("mydb3.db")
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            col_names = [desc[0] for desc in cursor.description]

            results = [dict(zip(col_names, row)) for row in rows]
            conn.close()
            return results
        except Exception as e:
            return f"Error executing SQL: {e}. The query might be invalid."

    def _format_response(self, original_query: str, results: str) -> str:
        if not results or results == "[]":
            return f"I looked in the database for '{original_query}', but I couldn't find any information."

        prompt = ChatPromptTemplate.from_template("""
You are a helpful customer support assistant. Based on the user's question and the data from the database, provide a clear and friendly response.
User's question: "{original_query}"
Database results:
{results}
Helpful Response:""").format(original_query=original_query, results=results)
        
        try:
            return self.llm.invoke(prompt).content.strip()
        except Exception as e:
            return f"Error formatting the response. Raw data: {results}"

# -------------------------------
# Customer Agent (The primary orchestrator)
# -------------------------------
class CustomerAgent:
    def __init__(self, llm, router: AgentRouter):
        self.llm = llm
        self.router = router
        # UPDATED PROMPT: Added a stricter instruction to prevent conversational filler.
        self.system_prompt = ChatPromptTemplate.from_template("""
You are the lead e-commerce Customer Support Agent. Your job is to understand the user's request and delegate it to the correct specialist agent.

**Your Specialist Team:**
1.  **DBAgent**: For any questions about existing data like order status, product stock, price, or customer details.
2.  **BuyerAgent**: When the user wants to buy or order a new product.
3.  **RecommendationAgent**: When the user wants to compare products or asks for recommendations (e.g., "which is better?").

**Your Process:**
1.  Analyze the user's message.
2.  Choose the single best agent for the task.
3.  Respond ONLY with the valid JSON object. Do NOT include any introductory text, explanations, or markdown formatting like ```json.
4.  Always include the correct `intent` field.
5.  Infer intent from the users message.


**JSON Response Format:**
{{
    "action": "delegate",
    "recipient": "AGENT_NAME",
    "query": "A clear query for the specialist agent.",
    "intent": "one of [order_status, returns, product_details, general_inquiry, purchase_orders, order_purchases]"
}}

**Example Decisions:**
- User: "Where is my order O123?" -> {{"action": "delegate", "recipient": "DBAgent", "query": "What is the status of order O123?"}}
- User: "I want to buy a smartwatch." -> {{"action": "delegate", "recipient": "BuyerAgent", "query": "User wants to buy a smartwatch."}}
- User: "hello" -> {{"action": "delegate", "recipient": "CustomerAgent", "query": "Hello! How can I assist you today?"}}

**Conversation History:**
{history}
**User's Latest Message:**
{input}
""")

    # NEW HELPER FUNCTION: To extract JSON from the LLM's potentially messy response.
    def _extract_json_from_response(self, text: str) -> str:
        """
        Extracts a JSON object from a string, even if it's embedded in other text.
        """
        # First, try to find JSON within ```json ... ```
        match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            return match.group(1)

        # If no markdown block, find the first '{' and the last '}'
        try:
            start_index = text.index('{')
            end_index = text.rindex('}') + 1
            return text[start_index:end_index]
        except ValueError:
            # If no '{' or '}' is found, return an empty string
            return ""

    def run(self, conversation_history: List[Dict]) -> Dict:
        print("[CustomerAgent]: Analyzing user request...")
        
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])
        prompt = self.system_prompt.format(history=history_str, input=conversation_history[-1]['content'])

        try:
            raw_response = self.llm.invoke(prompt).content.strip()
            print(f"[CustomerAgent]: LLM Raw Response: {raw_response}")

            # Use the new helper function to get clean JSON
            clean_json_str = self._extract_json_from_response(raw_response)
            
            if not clean_json_str:
                # If no JSON could be extracted, handle the error
                raise ValueError("No JSON object found in the response.")

            decision = json.loads(clean_json_str)
            print(f"[CustomerAgent]: Successfully Parsed Decision: {decision}")

        except (json.JSONDecodeError, ValueError, Exception) as e:
            print(f"[CustomerAgent]: Error parsing LLM decision. Defaulting to direct response. Error: {e}")
            return {"content": "I'm sorry, I'm having a little trouble understanding. Could you please rephrase?", "next_agent": "CustomerAgent"}

        if decision.get("action") == "delegate":
            recipient = decision.get("recipient")
            if recipient == "CustomerAgent": # Handle direct responses
                return {"content": decision.get("query"), "next_agent": "CustomerAgent"} 
            return self.router.route_message(sender="CustomerAgent",recipient=recipient,query=decision.get("query"),conversation_history=conversation_history,intent=decision.get("intent", "general_inquiry"))
        else:
            return {"content": "I'm not sure how to handle that request. Please try again.", "next_agent": "CustomerAgent"}

# at bottom of main11.py
agent_router = AgentRouter()
def run_pipeline(session_id: str, user_input: str, conversation_history: list,last_candidates=None):
    global agent_router  

    active_agent_name = conversation_history[-1]["next_agent"] if conversation_history else "CustomerAgent"
    active_agent = agent_router.get_agent(active_agent_name)

    if active_agent_name == "CustomerAgent":
        response_data = active_agent.run(
            conversation_history + [{"role": "user", "content": user_input, "session_id": session_id}]
        )
    else:
        response_data = active_agent.handle_request(
            sender="User",
            query=user_input,
            conversation_history=conversation_history + [{"role": "user", "content": user_input, "session_id": session_id}],
            intent="general_inquiry"
        )

    final_answer = response_data.get("content", "Sorry, I encountered an issue.")
    next_agent = response_data.get("next_agent", "CustomerAgent")

    return {"final_answer": final_answer, "next_agent": next_agent}

# -------------------------------
# Main Conversational Loop
# -------------------------------
# if __name__ == "__main__":
#     print("Starting E-Commerce Support Bot...")
#     agent_router = AgentRouter()
#     session_id = CommonUtils.get_new_session_id()
#     print(f"New conversation started. Session ID: {session_id}")

#     conversation_history = []
#     active_agent_name = "CustomerAgent"

#     while True:
#         user_input = input("You: ")
#         if user_input.lower() in ["exit", "quit"]:
#             print("Ending conversation. Goodbye!")
#             break

#         response = run_pipeline(session_id, user_input, conversation_history)
#         print(f"Agent: {response['final_answer']}")

# if __name__ == "__main__":
#     print("Starting E-Commerce Support Bot...")
#     agent_router = AgentRouter()
#     session_id = CommonUtils.get_new_session_id()
#     print(f"New conversation started. Session ID: {session_id}")

#     conversation_history = []
#     active_agent_name = "CustomerAgent" # Start with the CustomerAgent

#     while True:
#         user_input = input("You: ")
#         if user_input.lower() in ["exit", "quit"]:
#             print("Ending conversation. Goodbye!")
#             break

#         conversation_history.append({"role": "user", "content": user_input, "session_id": session_id})
#         CommonUtils.log_conversation(session_id, "user", user_input)

#         active_agent = agent_router.get_agent(active_agent_name)
        
#         if active_agent_name == "CustomerAgent":
#             response_data = active_agent.run(conversation_history)
#         else: # If another agent is active (e.g., BuyerAgent), route the new message directly to it
#              response_data = active_agent.handle_request(
#                 sender="User",
#                 query=user_input, # The agent will use the content from history
#                 conversation_history=conversation_history
#             )

#         final_answer = response_data.get("content", "Sorry, I encountered an issue.")
#         active_agent_name = response_data.get("next_agent", "CustomerAgent") # Update the active agent for the next turn

#         conversation_history.append({"role": "agent", "content": final_answer, "session_id": session_id})
#         CommonUtils.log_conversation(session_id, "agent", final_answer)

#         print(f"Agent: {final_answer}")
