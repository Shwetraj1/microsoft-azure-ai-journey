import streamlit as st
import os
from dotenv import load_dotenv
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents.models import ListSortOrder

# Load environment variables
load_dotenv()
PROJECT_ENDPOINT = os.getenv("PROJECT_ENDPOINT")
AGENT_ID = os.getenv("AGENT_ID")

# Initialize Azure Project Client (cached)
@st.cache_resource
def get_agent_client():
    return AIProjectClient(
        credential=DefaultAzureCredential(
            exclude_environment_credential=True,
            exclude_managed_identity_credential=True,
        ),
        endpoint=PROJECT_ENDPOINT,
    )

project_client = get_agent_client()

# Streamlit UI Config
st.set_page_config(page_title="Azure Expense Agent", layout="centered")
st.title("💼 Azure AI Expense Claim Agent")

# Initialize Session State
if "thread_id" not in st.session_state:
    thread = project_client.agents.threads.create()
    st.session_state.thread_id = thread.id
    st.session_state.messages = []

st.write(f"**Thread ID:** `{st.session_state.thread_id}`")

# Display existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input
user_input = st.chat_input("Enter your message...")

if user_input:
    # Add user message to UI
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Send message to agent
    project_client.agents.messages.create(
        thread_id=st.session_state.thread_id,
        role="user",
        content=user_input
    )

    # Run the agent
    run = project_client.agents.runs.create_and_process(
        thread_id=st.session_state.thread_id,
        agent_id=AGENT_ID
    )

    if run.status == "failed":
        st.error(f"Error: {run.last_error}")

    # Retrieve all messages
    api_messages = project_client.agents.messages.list(
        thread_id=st.session_state.thread_id,
        order=ListSortOrder.ASCENDING
    )

    # Extract the latest assistant message
    for m in api_messages:
        if m.role == "agent" and m.text_messages:
            assistant_reply = m.text_messages[-1].text.value

    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_reply}
    )

    # Display latest assistant reply
    with st.chat_message("assistant"):
        st.markdown(assistant_reply)
