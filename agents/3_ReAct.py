"""
Agent 3: Robust AI agent

Objective:
1. Learn how to create TOOLS.
2. Create a ReAct graph.
3. Work with different types of graphs.
4. Test the robustness of our agent.
"""

from typing import Annotated, TypedDict, Sequence # Annotated adds metadata to type
from langchain_core.messages import BaseMessage # foundational class for all message types
from langchain_core.messages import ToolMessage # passes data back to LLM after a tool is called
from langchain_core.messages import SystemMessage # provides instructions to the LLM
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI 
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages # reducer function, rules for merging new data into current state
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def add(a: int, b: int):
    """Addition function that adds two numbers""" # docstring is necessary
    return a + b

@tool
def subtract(a: int, b: int):
    """Subtraction function that subtracts two numbers"""
    return a - b

@tool
def multiply(a: int, b: int):
    """Multiplication function that multiplies two numbers"""
    return a * b

tools = [add, subtract, multiply] # list of tools

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash").bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    sys_prompt = SystemMessage(content=
        "You are my AI Agent, please answer my query to the best of your ability."
    )
    response = model.invoke([sys_prompt] + list(state["messages"])) 
    return {"messages": [response]} # add_messages will handle the appending for us


def should_cont(state: AgentState):
    messages = state["messages"]
    last_msg = messages[-1]
    if not last_msg.tool_calls: # type: ignore
        return "end"
    else: return "continue"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("agent", model_call)

    tool_node = ToolNode(tools=tools) # a single node that contains all the tools
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_cont,
        {
            "end": END,
            "continue": "tools",
        }
    )
    graph.add_edge("tools", "agent")
    return graph



if __name__ == "__main__":
    graph = build_graph()
    app = graph.compile()

    input = {"messages": [("user", "Add 40 + 12 and then multiply the result by 6.")]}

    # .stream returns a generator that yields messages
    # stream_mode="values" gives full graph state
    stream = app.stream(input, stream_mode="values") 
    for s in stream:
        msg = s["messages"][-1]
        if isinstance(msg, tuple):
            print(msg)
        else:
            msg.pretty_print()

