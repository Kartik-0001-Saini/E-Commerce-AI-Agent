import sqlite3
import uuid
from datetime import datetime

DB_PATH = "mydb3.db"

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