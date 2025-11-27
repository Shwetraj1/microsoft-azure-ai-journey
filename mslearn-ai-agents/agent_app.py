import os
from dotenv import load_dotenv
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents.models import ListSortOrder

load_dotenv()

PROJECT_ENDPOINT = os.getenv("PROJECT_ENDPOINT")
AGENT_ID = os.getenv("AGENT_ID")

client = AIProjectClient(
    credential=DefaultAzureCredential(
        exclude_environment_credential=True,
        exclude_managed_identity_credential=True
    ),
    endpoint=PROJECT_ENDPOINT
)

def chat_with_agent(user_message):
    thread = client.agents.threads.create()
    print(f"\nThread created: {thread.id}")

    client.agents.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_message
    )

    run = client.agents.runs.create_and_process(
        thread_id=thread.id,
        agent_id=AGENT_ID
    )

    if run.status == "failed":
        print("Run failed:", run.last_error)
        return

    messages = client.agents.messages.list(
        thread_id=thread.id,
        order=ListSortOrder.ASCENDING
    )

    print("\nAssistant Response:\n")
    for m in messages:
        if m.text_messages:
            print(f"{m.role}: {m.text_messages[-1].text.value}")

if __name__ == "__main__":
    while True:
        text = input("\nYou: ")
        if text.lower() in ["quit", "exit"]:
            break
        chat_with_agent(text)
