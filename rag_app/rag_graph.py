from langchain import hub
from langchain.schema import Document
from langgraph.graph import StateGraph, START
from typing import Any, List, TypedDict

from ingest.vector_store import ChromaVectorStore

# ---------------------------
# Define state for langgraph
# ---------------------------
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# ---------------------------
# Define prompt for question-answering
# ---------------------------
prompt = hub.pull("rlm/rag-prompt")

# ---------------------------
# Build graph steps
# ---------------------------
def retrieve(
    state: State,
    vector_store: ChromaVectorStore
):
    """
    Retrieve relevant documents using similarity search.
    
    Args:
        state: Current state
        vector_store: Vector store instance
    """
    docs = vector_store.similarity_search(state["question"], k=3)
    return {"context": docs}

def generate(state: State, llm: Any):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    # Handle different response types safely
    answer = response.content if hasattr(response, 'content') else str(response)
    return {"answer": answer}

def build_langgraph(
    vector_store: ChromaVectorStore,
    llm: Any
):
    """
    Build LangGraph for RAG pipeline.
    
    Args:
        vector_store: Vector store instance
        llm: LLM instance
    """
    # Wrap steps to pass vector_store and llm
    def retrieve_step(state: State):
        return retrieve(state, vector_store)

    def generate_step(state: State):
        return generate(state, llm)

    graph_builder = StateGraph(State).add_sequence([retrieve_step, generate_step])
    graph_builder.add_edge(START, "retrieve_step")
    return graph_builder.compile()