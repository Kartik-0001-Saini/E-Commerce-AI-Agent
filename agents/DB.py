import sqlite3
from typing import Dict, List, Union

# from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from utils.Commonutils import CommonUtils, DB_PATH

class DBAgent:
    def __init__(self, llm):
        self.llm = llm
        self.db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

    def handle_request(self, sender: str, query: str, conversation_history: List[Dict], intent: str="general_inquiry") -> Dict:
        print(f"[DBAgent]: Received request from {sender}: '{query}' with intent: {intent}")
        sql_query = self._generate_sql(query, intent)
        
        if "Error:" in sql_query or "unsafe" in sql_query.lower():
            return {"content": sql_query, "next_agent": "CustomerAgent"}

        results = self._execute_sql(sql_query)
        
        if isinstance(results, str) and "Error:" in results:
            return {"content": results, "next_agent": "CustomerAgent"}

        response = self._format_response(original_query=query, results=results)
        return {"content": response, "next_agent": "CustomerAgent"}

    def _generate_sql(self, question: str, intent: str) -> str:
        schema = self.db.get_table_info()
        sql_prompt_intent = ChatPromptTemplate.from_template("""
        You are an SQL expert for an e-commerce support chatbot.
        Database Schema: {schema}
        Rules:
        - Only output a single SELECT statement. No explanations.
        - The intent is: {intent}.
        - Always join products with suppliers using supplier_id when needed.
        Extra rules by intent:
        - for order_status -> select: status, tracking_number, created_at
        - for returns -> select: status, reason, created_at
        - for product_details -> select: product_id, name, price, stock, description
        - for general_inquiry (suppliers) -> select: supplier_id, name, product_name, warehouse_location, feature, discount_value
        User asked: {question}
        SQL Query:
        """)
        
        sql_raw = self.llm.invoke(sql_prompt_intent.format(
            schema=schema, intent=intent, question=question
        )).content.strip()
        return CommonUtils.clean_sql(sql_raw)

    def _execute_sql(self, sql: str) -> Union[List[Dict], str]:
        if any(word in sql.upper() for word in ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER"]):
            return "Error: Unsafe SQL detected. Only SELECT queries are allowed."
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            col_names = [desc[0] for desc in cursor.description]
            results = [dict(zip(col_names, row)) for row in rows]
            conn.close()
            return results
        except Exception as e:
            return f"Error executing SQL: {e}. The query might be invalid."

    def _format_response(self, original_query: str, results: List[Dict]) -> str:
        if not results:
            return f"I looked in the database for '{original_query}', but I couldn't find any information."

        prompt = ChatPromptTemplate.from_template("""
        You are a helpful customer support assistant. Based on the user's question and the data, provide a clear and friendly response.
        User's question: "{original_query}"
        Database results: {results}
        Helpful Response:
        """).format(original_query=original_query, results=results)
        
        try:
            return self.llm.invoke(prompt).content.strip()
        except Exception as e:
            return f"Error formatting the response. Raw data: {results}"
