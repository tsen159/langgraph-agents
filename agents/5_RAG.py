"""
What you can ask:
- How was SMP500 performance in 2024?
    (This will return the relevant chunks from the PDF document.)
- How did OpenAI perform in 2024?
    (This is not in the document.)
"""


import os
from typing import Annotated, TypedDict, Sequence 
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma # lightweight vector store
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages 
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()
dir_name = os.path.dirname(__file__)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
)

# embedding model has to be compatible with the LLM model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
)

pdf_path = f"{dir_name}/Stock_Market_Performance_2024.pdf"
pdf_loader = PyPDFLoader(pdf_path)


# Checks if the PDF is there
try:
    pages = pdf_loader.load()
    print(f"PDF has been loaded and has {len(pages)} pages")
except Exception as e:
    print(f"Error loading PDF: {e}")
    raise

# Chunking process
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
pages_split = text_splitter.split_documents(pages)

# Create a vector store from the chunks
persist_directory = dir_name
collection_name = "stock_market"
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

try:
    # create the chroma database using our embedding model
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print(f"Created ChromaDB vector store")
    
except Exception as e:
    print(f"Error setting up ChromaDB: {str(e)}")
    raise

# create a retriever to retrieve relevant chunks
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5} # K is the amount of chunks to return
)


@tool
def retriever_tool(query: str) -> str:
    """
    Retrieves relevant chunks from the vector store based on the query.
    
    Args:
        query: The query to search for in the vector store.
    """
    
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found in the document."
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Doc {i+1}:\n{doc.page_content}\n")
    return "\n".join(results)
        
tools = [retriever_tool]  
llm = llm.bind_tools(tools)
tools_dict = {our_tool.name: our_tool for our_tool in tools}

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]  


system_prompt = """
    You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 /
    based on the PDF document loaded into your knowledge base.
    Use the retriever tool available to answer questions about the stock market performance data. /
    You can make multiple calls if needed.
    If you need to look up some information before asking a follow up question, you are allowed to do that!
    Please always cite the specific parts of the documents you use in your answers.
"""

# agent
def llm_agent(state: AgentState) -> AgentState:
    """This node will answer questions based on the provided messages and the retriever tool."""
    sys_prompt = SystemMessage(content=system_prompt)
    response = llm.invoke([sys_prompt] + list(state["messages"])) 
    return {"messages": [response]} # add_messages will handle the appending for us


# retriever agent
def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""

    tool_calls = state['messages'][-1].tool_calls # type: ignore
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
        
        if not t['name'] in tools_dict: # Checks if a valid tool is present
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")
            

        # Appends the Tool Message
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("Tools Execution Complete. Back to the model!")
    return {'messages': results}



# function
def should_continue(state: AgentState):
    """Check if the last message is a tool call or not."""
    msg = state["messages"][-1]
    if hasattr(msg, "tool_calls") and len(msg.tool_calls) > 0: # type: ignore
        return "continue"
    else:
        return "end"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("llm", llm_agent)
    graph.add_node("retriever_agent", take_action)
    
   

    graph.add_edge(START, "llm")
    graph.add_conditional_edges(
        "llm",
        should_continue,
        {
            "end": END,
            "continue": "retriever_agent",
        }
    )
    
    graph.add_edge("retriever_agent", "llm")  # Go back to the LLM after tool execution
    return graph


def running_agent(rag_agent):
    print("\n=== RAG AGENT===")
    
    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        messages = [HumanMessage(content=user_input)] # converts back to a HumanMessage type

        result = rag_agent.invoke({"messages": messages})
        
        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)


if __name__ == "__main__":
    graph = build_graph()
    rag_agent = graph.compile()
    running_agent(rag_agent)


