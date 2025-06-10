"""
Exercise 3: 
1. Accept a user's name, age, and a list of their skills.
2. Pass the state throught three nodes that:
    First: Personalizes the name field with a greeting.
    Second: Describe the user's age.
    Third: Lists the user's skills in a formatted string.
3. The final output in the result field should be a combined message:
    Output: "Linda, welcome! You are 31 years old. You have skills in: Python, Machine Learning, and LangGraph.
"""

from typing import List, TypedDict
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    name: str
    age: int
    skills: List[str]
    result: str

def node_1(state: AgentState) -> AgentState:
    state["result"] = f"{state["name"]}, welcome!"
    return state
def node_2(state: AgentState) -> AgentState:
    state["result"] = state["result"] + f" You are {state["age"]} years old."
    return state
def node_3(state: AgentState) -> AgentState:
    skill_str = ""
    skill_list = state["skills"]
    for i in range(len(skill_list)):
        if i == len(skill_list) - 1:
            skill_str += f"and {skill_list[i]}."
        else: skill_str += f"{skill_list[i]}, "

    state["result"] = state["result"] + f" You have skills in: {skill_str}"
    return state

graph = StateGraph(AgentState)
graph.add_node("node_1", node_1)
graph.add_node("node_2", node_2)
graph.add_node("node_3", node_3)

graph.set_entry_point("node_1")
graph.add_edge("node_1", "node_2")
graph.add_edge("node_2", "node_3")
graph.set_finish_point("node_3")
app = graph.compile()

if __name__ == "__main__":
    input = {"name": "Linda", "age": 31, "skills": ["Python", "Machine Learning", "LangGraph"]}
    output = app.invoke(input)
    print(output["result"])