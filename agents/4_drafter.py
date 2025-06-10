"""
Agent 4: AI agent that drafts documents

Note: 
Tool node connects to the END node, since we need to save the document after drafting.
"""
import os
from typing import Annotated, TypedDict, Sequence # Annotated adds metadata to type
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI 
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages # reducer function, rules for merging new data into current state
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()
dir_name = os.path.dirname(__file__)

# a global variable to store the document
document_content = ""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]  

@tool
def update(content: str) -> str: # content will be provided by the LLM in the background
    """Updates the document with the provided content."""
    global document_content
    document_content = content
    print(f"\n📄 Document updated:\n{document_content}\n")
    return "Document updated."

@tool
def save(filename: str) -> str:
    """
    Saves the document to a file.

    Args:
        filename: The name of the text file.
    """
    global document_content
    if not filename.endswith(".txt"): # ensure the filename ends with .txt
        filename += ".txt"
    filename = os.path.join(dir_name, filename)  # save in the same directory as this script

    # save
    try:
        with open(filename, "w") as file:
            file.write(document_content)
        return f"Document saved to {filename}."
    except Exception as e: 
        return f"Error saving document: {str(e)}"
    
tools = [update, save]  
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash").bind_tools(tools)

def agent(state: AgentState) -> AgentState:
    """
    This node will draft the document based on the provided messages.
    """
    sys_prompt = SystemMessage(content=f"""
        You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
        - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
        - If the user wants to save and finish, you need to use the 'save' tool.
        - Make sure to always show the current document state after modifications.
        
        The current document content is:{document_content}
        """
    )

    if not state["messages"]: # work as a initial message if no messages are provided
        user_input = "I would like to draft a document. Please help me with that."
        user_message = HumanMessage(content=user_input)

    else:
        user_input = input("\nWhat would you like to do with the document? ")
        print(f"\n👤 USER: {user_input}")
        user_message = HumanMessage(content=user_input)
    
    messages = [sys_prompt] + list(state["messages"]) + [user_message] 
    response = model.invoke(messages) 

    print(f"\n🤖 AI: {response.content}") 
    if hasattr(response, "tool_calls") and response.tool_calls: # type: ignore
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}") # type: ignore
    return {"messages": list(state["messages"]) + [user_message, response]} 


 
def should_continue(state: AgentState):
    """Determines whether to continue drafting or end the process."""
    messages = state["messages"]
    if not messages: return "continue"

    # looks for most recent tool message
    for message in reversed(messages):
        # check if it is a ToolMessage resulting from save
        if (isinstance(message, ToolMessage) and
            "saved" in message.content.lower() and # type: ignore
            "document" in message.content.lower()): # type: ignore
            return "end"
            
    return "continue"

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)

    tool_node = ToolNode(tools=tools)  # a single node that contains all the tools
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_edge("agent", "tools")
    graph.add_conditional_edges(
        "tools",
        should_continue,
        {
            "end": END,
            "continue": "agent",
        }
    )
    
    return graph

if __name__ == "__main__":
    graph = build_graph()
    app = graph.compile()

    print("\n ===== DRAFTER =====")
    
    state = {"messages": []}
    
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            messages = step["messages"]
            if not messages:
                continue
    
            for message in messages[-3:]:
                if isinstance(message, ToolMessage):
                    print(f"\n🛠️ TOOL RESULT: {message.content}")
    
    print("\n ===== DRAFTER FINISHED =====")
