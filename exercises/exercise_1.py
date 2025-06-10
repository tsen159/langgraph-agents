"""
Exercise 1: Create a personal compliment Agent using LangGraph.

Input: {"name": "Bob"}
Output: "Bob, you are doing great!"
"""

from typing import Dict, TypedDict
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    message: str

def process(state: AgentState) -> AgentState:
    state["message"] = f"{state['message']}, you are doing great!"
    return state

# create graph
graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.set_entry_point("process")
graph.set_finish_point("process")
app = graph.compile()

if __name__ == "__main__":
    # Test the agent with an example input
    input_data: Dict[str, str] = {"message": "Bob"}
    output = app.invoke(input_data)
    print(output["message"])  # Expected output: "Bob, you are doing great!"