import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import ConnectedAgentTool, ListSortOrder

load_dotenv()

PROJECT_ENDPOINT = os.getenv("PROJECT_ENDPOINT")
MODEL_DEPLOYMENT = os.getenv("MODEL_DEPLOYMENT")

agents_client = AgentsClient(
    endpoint=PROJECT_ENDPOINT,
    credential=DefaultAzureCredential(
        exclude_environment_credential=True,
        exclude_managed_identity_credential=True
    )
)

# -------- PRIORITY AGENT --------
priority_agent = agents_client.create_agent(
    model=MODEL_DEPLOYMENT,
    name="priority_agent",
    instructions="""
Assess how urgent a ticket is based on its description.

Respond with:
- High
- Medium
- Low
Plus a brief explanation.
"""
)

# -------- TEAM AGENT --------
team_agent = agents_client.create_agent(
    model=MODEL_DEPLOYMENT,
    name="team_agent",
    instructions="""
Decide which team should own each ticket.

Teams:
- Frontend
- Backend
- Infrastructure
- Marketing

Respond with team and brief explanation.
"""
)

# -------- EFFORT AGENT --------
effort_agent = agents_client.create_agent(
    model=MODEL_DEPLOYMENT,
    name="effort_agent",
    instructions="""
Estimate required effort:

- Small (1 day)
- Medium (2-3 days)
- Large (multiple days)

Respond with level and brief justification.
"""
)

# -------- CONNECT TOOLS --------
priority_tool = ConnectedAgentTool(
    id=priority_agent.id,
    name="priority_agent",
    description="Determines ticket urgency"
)

team_tool = ConnectedAgentTool(
    id=team_agent.id,
    name="team_agent",
    description="Assigns the team"
)

effort_tool = ConnectedAgentTool(
    id=effort_agent.id,
    name="effort_agent",
    description="Estimates effort"
)

# -------- TRIAGE MASTER AGENT --------
triage_agent = agents_client.create_agent(
    model=MODEL_DEPLOYMENT,
    name="triage_agent",
    instructions="""
You are responsible for triaging support tickets.
Use the connected agents to determine:
1. Priority
2. Assigned Team
3. Estimated Effort
Present the final structured result.
""",
    tools=[
        priority_tool.definitions[0],
        team_tool.definitions[0],
        effort_tool.definitions[0]
    ]
)

# -------- RUN SYSTEM --------
thread = agents_client.threads.create()

ticket = input("Describe the support ticket: ")

agents_client.messages.create(
    thread_id=thread.id,
    role="user",
    content=ticket
)

run = agents_client.runs.create_and_process(
    thread_id=thread.id,
    agent_id=triage_agent.id
)

messages = agents_client.messages.list(thread_id=thread.id, order=ListSortOrder.ASCENDING)

print("\n--- FINAL TRIAGE REPORT ---\n")
for m in messages:
    if m.text_messages:
        print(m.text_messages[-1].text.value)
