"""
Agent 1: Simple Bot

Goal: 
Integrate LLMs in our graph.
(Since there's no memory, llm cannot access to old conversations.)
"""

from typing import List, TypedDict
from langchain_core.messages import HumanMessage
# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI 
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv # used to store secret stuffs like API key

load_dotenv() # load your personal API key

class AgentState(TypedDict):
    message: List[HumanMessage] # mention it's a human message type

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# node
def process_node(state: AgentState) -> AgentState:
    response = llm.invoke(state["message"]) # llm reads message
    print(f"AI: {response.content}")
    return state


graph = StateGraph(AgentState)
graph.add_node("process", process_node)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

user_input = input("Enter: ")
while user_input != "exit":
    agent.invoke(
        {"message": [HumanMessage(content=user_input)]}
    )
    user_input = input("Enter: ")