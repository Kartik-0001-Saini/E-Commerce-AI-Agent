import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from agents.Router import AgentRouter # <-- Import the router from its new location
import time 
import mlflow

# --- Environment & LLM Setup ---
load_dotenv()
llm_db = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.getenv("key_sal_groq"))
llm_customer = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.getenv("key_jay_groq"))
llm_reco = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.getenv("key_veer_groq"))

# --- Singleton Instance of the Router ---
# This ensures we only have one instance of the agents running
agent_router = AgentRouter(llm_customer=llm_customer, llm_db=llm_db, llm_reco=llm_reco)

# --- Pipeline Execution ---
def run_pipeline(session_id: str, user_input: str, conversation_history: list, mlflow_run_id: str, last_candidates=None):
    global agent_router
    start_time = time.time()

    # No need to search for the run anymore, we have the ID.
    # We establish the context of the parent run, then create a nested run inside it.
    with mlflow.start_run(run_id=mlflow_run_id):
        # ✅ Create a nested run for this specific turn
        with mlflow.start_run(nested=True, run_name=f"Turn_{len(conversation_history)}") as turn_run:
            print(f"[MLflow]: Started nested run {turn_run.info.run_id} for turn {len(conversation_history)}")

            # --- Log turn-specific parameters ---
            # Now it's safe to log these for every turn
            active_agent_name = conversation_history[-1].get("next_agent", "CustomerAgent")
            mlflow.log_param("active_agent", active_agent_name)
            mlflow.log_param("user_input", user_input)

            # Restore last candidates if they exist
            if last_candidates:
                agent_router.reco_agent.last_candidates = last_candidates

            active_agent = agent_router.get_agent(active_agent_name)
            
            current_turn_history = conversation_history + [{"role": "user", "content": user_input, "session_id": session_id}]

            if active_agent_name == "CustomerAgent":
                response_data = active_agent.run(current_turn_history)
            else:
                response_data = active_agent.handle_request(
                    sender="User",
                    query=user_input,
                    conversation_history=current_turn_history,
                    intent="general_inquiry"
                )

            final_answer = response_data.get("content", "Sorry, I encountered an issue.")
            next_agent = response_data.get("next_agent", "CustomerAgent")
            
            # --- Log turn-specific metrics and params ---
            end_time = time.time()
            response_time = end_time - start_time
            mlflow.log_metric("response_time_seconds", response_time)
            mlflow.log_param("next_agent_determined", next_agent)
            
            # Log the agent's response as a turn-specific artifact
            mlflow.log_text(final_answer, "agent_response.txt")

    # Prepare the final response dictionary (this is outside the mlflow blocks)
    final_response = {
        "final_answer": final_answer,
        "next_agent": next_agent,
        "last_candidates": agent_router.reco_agent.last_candidates
    }
    
    return final_response