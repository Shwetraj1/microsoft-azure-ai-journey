import os
import streamlit as st
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from openai import AzureOpenAI  # IMPORTANT


# Load environment variables
load_dotenv()
project_endpoint = os.getenv("PROJECT_ENDPOINT")
model_deployment = os.getenv("MODEL_DEPLOYMENT")


# Streamlit page configuration
st.set_page_config(page_title="Azure OpenAI Chat", layout="centered")
st.title("🤖 Azure OpenAI Chat Assistant")


# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "system", "content": "You are a helpful AI assistant that answers questions."}
    ]


# Initialize Azure AI Project client and OpenAI client
@st.cache_resource
def get_clients():
    try:
        project_client = AIProjectClient(
            credential=DefaultAzureCredential(
                exclude_environment_credential=True,
                exclude_managed_identity_credential=True
            ),
            endpoint=project_endpoint,
        )

        # NEW: Correct way to get an OpenAI client from Azure AI Foundry
        openai_client = project_client.get_openai_client(api_version="2024-10-21")

        return openai_client

    except Exception as e:
        st.error(f"Error initializing clients: {e}")
        return None


client = get_clients()


# Chat input
user_input = st.chat_input("Enter a prompt...")


# Handle chat interaction
if user_input and client:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    try:
        response = client.chat.completions.create(
            model=model_deployment,
            messages=st.session_state.chat_history
        )
        assistant_reply = response.choices[0].message.content
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})
    except Exception as e:
        st.error(f"API Error: {e}")


# Display chat history
for msg in st.session_state.chat_history[1:]:  # Skip the initial system message
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
