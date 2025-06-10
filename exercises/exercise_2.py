"""
Exercise 2: 
Create a graph where you pass in a single list of integers along with a name and a operation.
Add the elements if the operation is "+",
and multiply them if the operation is "*",
all within the same node.

Input: {"name": "Jack", "values": [1, 2, 3, 4], "operation": "*"} 
Output: "Jack, the result is 6!"
"""

from typing import List, TypedDict
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    name: str
    values: List[int]
    operation: str
    result: str


def process_node(state: AgentState) -> AgentState:
    if state["operation"] == "+":
        total = sum(state["values"])
    elif state["operation"] == "*":
        total = 1
        for value in state["values"]:
            total *= value
    state["result"] = f"{state['name']}, the result is {total}!"

    return state

# create graph
graph = StateGraph(AgentState)
graph.add_node("process_node", process_node)
graph.set_entry_point("process_node")
graph.set_finish_point("process_node")
app = graph.compile()

if __name__ == "__main__":
    input = {"name": "Jack", "values": [1, 2, 3, 4], "operation": "*"} 
    output = app.invoke(input)
    print(output["result"])


