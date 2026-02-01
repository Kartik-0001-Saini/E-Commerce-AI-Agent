# agents/Buyer.py

import sqlite3
import uuid
from datetime import datetime
from langchain_groq import ChatGroq
import os

# Assuming RecommendationAgent is in a file that can be imported
# If not, you may need to pass a product fetching function to this class.
from .Recommendation import RecommendationAgent 

DB_PATH = "mydb3.db"

class BuyerAgent:
    def __init__(self, llm):
        self.llm = llm
        self.purchase_sessions = {} # In-memory state for ongoing purchases

    def handle_request(self, sender: str, query: str, conversation_history: list, intent: str = None) -> dict:
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
        transportation_cost = base_cost * 0.12  # fixed example, can make dynamic
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

    def _init_purchase_state(self, session_id: str) -> dict:
        return {"session_id": session_id, "purchase_stage": None, "purchase_candidates": None, "selected_product": None, "total_units": None, "buyer_name": None, "buyer_location": None, "buyer_contact": None, "summary": None}

    def _get_product_keyword(self, text: str) -> str:
        try:
            prompt = f"""Extract the main product keyword from: "{text}". Return only one word or phrase."""
            return self.llm.invoke(prompt).content.strip().lower()
        except Exception:
            return text.split()[-1]

    def _present_options_and_update(self, state: dict):
        user_text = state.get("question", "")
        keyword = self._get_product_keyword(user_text)
        # We instantiate RecommendationAgent here just to use its fetch_products utility.
        # This creates a dependency, which is okay for now.
        llm_reco = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.getenv("key_veer_groq"))
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
        return "Purchase flow is in progress."