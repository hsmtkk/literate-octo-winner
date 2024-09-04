import operator
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from langchain_core.agents import AgentAction, AgentFinish
from langgraph.prebuilt import ToolExecutor
from langgraph.graph import StateGraph,END

llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)
tools = [TavilySearchResults(max_results=10)]
prompt = hub.pull("hwchase17/openai-functions-agent")
agent_runnable = create_openai_functions_agent(llm, tools, prompt)

class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    agent_outcome: AgentAction | AgentFinish | None
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

tool_executor = ToolExecutor(tools)

def run_agent(state: AgentState) -> AgentState:
    agent_outcome = agent_runnable.invoke(state)
    return {"agent_outcome": agent_outcome}

def execute_tools(state: AgentState) -> AgentState:
    agent_action = state["agent_outcome"]
    output = tool_executor.invoke(agent_action)
    return {"intermediate_steps": [(agent_action, str(output))]}

def should_continue(state: AgentState) -> str:
    if isinstance(state["agent_outcome"], AgentFinish):
        return "end"
    else:
        return "continue"

workflow = StateGraph(AgentState)
workflow.add_node("agent", run_agent)
workflow.add_node("action", execute_tools)
workflow.set_entry_point("agent")

workflow.add_conditional_edges("agent", should_continue, {"continue": "action", "end": END})
workflow.add_edge("action", "agent")
app = workflow.compile()

inputs = {"input":"Can you recommend a good Italian restaurant in New York City?", "chat_history": []}

for s in app.stream(inputs):
    print(list(s.values())[0])
    print("---")