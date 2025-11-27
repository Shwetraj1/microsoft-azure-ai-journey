import os
from dotenv import load_dotenv
from openai import AzureOpenAI
import streamlit as st


# --- MUST be the first Streamlit command ---
st.set_page_config(page_title="Margie's Travel Assistant", page_icon="✈️")


# Load environment variables
load_dotenv()


OPEN_AI_ENDPOINT = os.getenv("OPEN_AI_ENDPOINT")
OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
SEARCH_ENDPOINT = os.getenv("SEARCH_ENDPOINT")
SEARCH_KEY = os.getenv("SEARCH_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")


# Initialize chat client once
@st.cache_resource
def get_chat_client():
    return AzureOpenAI(
        api_version="2024-12-01-preview",
        azure_endpoint=OPEN_AI_ENDPOINT,
        api_key=OPEN_AI_KEY
    )


chat_client = get_chat_client()


def get_rag_params():
    return {
        "data_sources": [
            {
                "type": "azure_search",
                "parameters": {
                    "endpoint": SEARCH_ENDPOINT,
                    "index_name": INDEX_NAME,
                    "authentication": {
                        "type": "api_key",
                        "key": SEARCH_KEY,
                    },
                    "query_type": "vector",
                    "embedding_dependency": {
                        "type": "deployment_name",
                        "deployment_name": EMBEDDING_MODEL,
                    },
                }
            }
        ],
    }


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "system",
                "content": "You are a travel assistant that provides information on travel services available from Margie's Travel."
            }
        ]


def main():
    st.title("✈️ Margie's Travel Assistant")
    st.write("Ask anything about travel services available from Margie's Travel.")


    # Sidebar config
    with st.sidebar:
        st.header("Configuration")
        st.text(f"Chat model: {CHAT_MODEL or 'Not set'}")
        st.text(f"Embedding model: {EMBEDDING_MODEL or 'Not set'}")
        st.text(f"Search index: {INDEX_NAME or 'Not set'}")


    init_session_state()


    # Display chat messages
    for msg in st.session_state.messages:
        if msg["role"] != "system":
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])


    # User input box
    user_input = st.chat_input("Enter your prompt")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})


        with st.chat_message("user"):
            st.markdown(user_input)


        try:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = chat_client.chat.completions.create(
                        model=CHAT_MODEL,
                        messages=st.session_state.messages,
                        extra_body=get_rag_params()
                    )
                    reply = response.choices[0].message.content
                    st.markdown(reply)


            st.session_state.messages.append({"role": "assistant", "content": reply})


        except Exception as ex:
            st.error(f"Error: {ex}")


if __name__ == "__main__":
    main()





