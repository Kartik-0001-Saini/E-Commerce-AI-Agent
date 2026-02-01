import json
import re
import sqlite3
from typing import Dict, List
# from langchain.prompts import ChatPromptTemplate
from utils.Commonutils import DB_PATH
from langchain_core.prompts import ChatPromptTemplate



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
