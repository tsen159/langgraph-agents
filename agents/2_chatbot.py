"""
Agent 2: A chatbot which has memory
"""

import os
from typing import List, TypedDict, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI 
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()
dir_name = os.path.dirname(__file__)

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]] # a list of HumanMessage or AIMessage

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

def process_node(state: AgentState) -> AgentState:
    """
    This node will solve the request you input.
    """
    response = llm.invoke(state["messages"]) # llm reads messages
    state["messages"].append(AIMessage(response.content))
    print(f"AI: {response.content}")
    return state


graph = StateGraph(AgentState)
graph.add_node("process", process_node)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

conversation_history = []

user_input = input("Enter: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(user_input))
    result = agent.invoke({"messages": conversation_history}) # send the entire conversation history
    conversation_history = result["messages"]
    user_input = input("Enter: ")

with open(f"{dir_name}/logging.txt", "w") as file: # save all messages on a database
    file.write("Your conversation log:\n")
    for msg in conversation_history:
        if isinstance(msg, HumanMessage):
            file.write(f"You: {msg.content}\n")
        elif isinstance(msg, AIMessage):
            file.write(f"AI: {msg.content}\n\n")
    file.write("End.")
print("Conversation saved to logging.txt.")


