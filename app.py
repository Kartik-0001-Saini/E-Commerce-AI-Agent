import streamlit as st
import sqlite3
import pandas as pd
from uuid import uuid4
import mlflow

# (Assuming your other imports like run_pipeline, etc., are here)
from pipeline.AGentic_pipeline import run_pipeline
from utils.Commonutils import CommonUtils


# -------------------------------
# Page Config & MLflow Setup
# -------------------------------
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

    # --- THIS BLOCK NOW HANDLES ALL ONE-TIME SETUP PER SESSION ---

    # 1. Start a new MLflow run when a new session is created
    run = mlflow.start_run(run_name=f"Session_{st.session_state['session_id']}")
    st.session_state["mlflow_run_id"] = run.info.run_id
    mlflow.set_tag("session_id", st.session_state["session_id"])

    # 2. Log the database as an input to THIS run (only happens once)
    try:
        db_file = "mydb3.db"
        # !!! IMPORTANT: Replace with your actual table names !!!
        table_names = ["employees", "orders", "customers","suppliers","purchase_orders"] # Example list of tables

        conn = sqlite3.connect(db_file)
        
        # Log each table as a separate dataset
        for table_name in table_names:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            dataset = mlflow.data.from_pandas(df, source=f"{db_file}/{table_name}", name=f"{table_name}_table")
            # Log the dataset as an input to the active run
            mlflow.log_input(dataset, context=f"database_snapshot_{table_name}")
        
        conn.close()
        print(f"Successfully logged database tables as datasets for run: {run.info.run_id}")

    except Exception as e:
        st.sidebar.error(f"Error logging database to MLflow: {e}")


if "next_agent" not in st.session_state:
    st.session_state["next_agent"] = "CustomerAgent"

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

# (The rest of your code for chat input and pipeline logic remains the same)
# ...
# -------------------------------
# Chat Input
# -------------------------------
if prompt := st.chat_input("Ask me about your orders, returns, or products..."):
    # If this is the first message of the session, log it as a parameter
    if not st.session_state["messages"]:
        # Use the existing run_id to log the parameter
        with mlflow.start_run(run_id=st.session_state["mlflow_run_id"], nested=True):
             mlflow.log_param("initial_user_prompt", prompt)

    st.session_state["messages"].append(("customer", prompt))
    st.chat_message("user").write(prompt)

    # ✅ Save to DB
    CommonUtils.log_conversation(
        st.session_state["session_id"], "customer", prompt
    )

    try:
        conversation_history = [
            { "role": "user" if r == "customer" else "agent", "content": m }
            for r, m in st.session_state["messages"]
        ]
        # 🚀 Call pipeline
        response = run_pipeline(
            st.session_state["session_id"],
            prompt,
            conversation_history,
            st.session_state["last_candidates"]
        )
        answer = response["final_answer"]
        st.session_state["next_agent"] = response["next_agent"]
        if "last_candidates" in response:
            st.session_state["last_candidates"] = response["last_candidates"]
    except Exception as e:
        answer = f"⚠️ Error: {e}"

    st.session_state["messages"].append(("agent", answer))
    st.chat_message("assistant").write(answer)
    CommonUtils.log_conversation(
        st.session_state["session_id"], "agent", answer
    )


# -------------------------------
# Sidebar and Clear Chat Logic
# -------------------------------

# (Your sidebar and clear chat logic can remain here)
# Small improvement to the "Clear Chat" button
if st.sidebar.button("🧹 Clear Chat & End Run"):
    # End the active MLflow run if one exists
    if "mlflow_run_id" in st.session_state and st.session_state["mlflow_run_id"]:
        try:
            # Log full conversation before ending
            history_str = "\n\n".join([f"{role.capitalize()}: {msg}" for role, msg in st.session_state["messages"]])
            with mlflow.start_run(run_id=st.session_state["mlflow_run_id"], nested=True):
                mlflow.log_text(history_str, "conversation_history.txt")
            mlflow.end_run()
        except Exception as e:
            st.sidebar.error(f"Failed to end MLflow run: {e}")

    # Clear all session state keys
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun()