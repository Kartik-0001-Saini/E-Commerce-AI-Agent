import streamlit as st
import sqlite3
from uuid import uuid4

from pipeline.AGentic_pipeline import run_pipeline
from agents.Router import AgentRouter
from utils.Commonutils import CommonUtils
# -------------------------------
# Page Config
# -------------------------------
import mlflow

# --- MLflow Setup ---
MLFLOW_TRACKING_URI = "http://127.0.0.1:8080"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("E-Commerce Chatbot Conversations")
st.set_page_config(page_title="E-Commerce AI Agent", page_icon="🛒", layout="wide")

st.title("🛒 E-Commerce Support Chatbot")
st.caption("Ask about orders, returns, products, or get recommendations.")

# -------------------------------
# Initialize Session State
# -------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "session_id" not in st.session_state:
    st.session_state["session_id"] = "S" + str(uuid4().hex[:6]).upper()
    # Start a new MLflow run when a new session is created
    run = mlflow.start_run(run_name=f"Session_{st.session_state['session_id']}")
    st.session_state["mlflow_run_id"] = run.info.run_id

if "next_agent" not in st.session_state:
    st.session_state["next_agent"] = "CustomerAgent"

# ✅ Store last product candidates (for compare queries)
if "last_candidates" not in st.session_state:
    st.session_state["last_candidates"] = []

# -------------------------------
# Show Chat History
# -------------------------------
for role, msg in st.session_state["messages"]:
    if role == "customer":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(msg)


import mlflow
import pandas as pd
import sqlite3

# Define the path to your database file
db_file = "mysql3.db"
table_name = "your_table_name"  # <--- Change this to your table name

# Start an MLflow run
with mlflow.start_run() as run:
    # Connect to the SQLite database
    conn = sqlite3.connect(db_file)

    # Read data from a table into a pandas DataFrame
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

    # Close the database connection
    conn.close()

    # Create an MLflow Dataset from the pandas DataFrame
    # The 'source' parameter records the origin of the data
    dataset = mlflow.data.from_pandas(df, source=db_file, name="agent_database")

    # Log the dataset as an input to the run
    mlflow.log_input(dataset, context="training")

    print(f"Logged '{db_file}' as a dataset in run: {run.info.run_id}")
# -------------------------------
# Chat Input
# -------------------------------
if prompt := st.chat_input("Ask me about your orders, returns, or products..."):
    # If this is the first message of the session, log it as a parameter
    if not st.session_state["messages"]:
        with mlflow.start_run(run_id=st.session_state["mlflow_run_id"]):
            mlflow.log_param("initial_user_prompt", prompt)
            mlflow.set_tag("session_id", st.session_state["session_id"])
    st.session_state["messages"].append(("customer", prompt))
    st.chat_message("user").write(prompt)

    # ✅ Save to DB
    CommonUtils.log_conversation(
        st.session_state["session_id"], "customer", prompt
    )

    try:
        # Build conversation history
        conversation_history = [
            {
                "role": "user" if r == "customer" else "agent",
                "content": m,
                "session_id": st.session_state["session_id"],
                "next_agent": st.session_state.get("next_agent", "CustomerAgent"),
            }
            for r, m in st.session_state["messages"]
        ]

        # 🚀 Call pipeline with last_candidates context
        response = run_pipeline(
            st.session_state["session_id"],
            prompt,
            conversation_history,
            st.session_state["last_candidates"]
        )

        answer = response["final_answer"]
        st.session_state["next_agent"] = response["next_agent"]
        # try:
        #     # Build a string representation of the conversation history
        #     history_str = ""
        #     for role, msg in st.session_state["messages"]:
        #         history_str += f"{role.capitalize()}: {msg}\n\n"

        #     # Log the full conversation as a text file artifact
        #     with mlflow.start_run(run_id=st.session_state["mlflow_run_id"]):
        #         mlflow.log_text(history_str, "conversation_history.txt")
                
        # except Exception as e:
        #     st.sidebar.error(f"Failed to log to MLflow: {e}")

        # ✅ Update last_candidates if returned
        if "last_candidates" in response:
            st.session_state["last_candidates"] = response["last_candidates"]

    except Exception as e:
        answer = f"⚠️ Error: {e}"

    # Log agent response
    st.session_state["messages"].append(("agent", answer))
    st.chat_message("assistant").write(answer)

    # ✅ Save to DB
    CommonUtils.log_conversation(
        st.session_state["session_id"], "agent", answer
    )

# -------------------------------
# Sidebar: Conversation History
# -------------------------------
def fetch_conversations(session_id):
    conn = sqlite3.connect("mydb3.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role, content, created_at FROM conversations WHERE session_id=? ORDER BY id",
        (session_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    return rows

st.sidebar.header("📜 Conversation History")

if st.sidebar.button("Show Current Session"):
    history = fetch_conversations(st.session_state["session_id"])
    if history:
        for role, content, created_at in history:
            if role == "customer":
                st.sidebar.write(f"🧑 **Customer** ({created_at}): {content}")
            else:
                st.sidebar.write(f"🤖 **Agent** ({created_at}): {content}")
    else:
        st.sidebar.info("No history found for this session.")

# -------------------------------
# Option: Clear Chat
# -------------------------------
if st.sidebar.button("🧹 Clear Chat"):
    st.session_state["messages"] = []
    st.session_state["next_agent"] = "CustomerAgent"
    st.session_state["last_candidates"] = []
    st.session_state["mlflow_run_id"]=[]
    mlflow.end_run()
    st.rerun()