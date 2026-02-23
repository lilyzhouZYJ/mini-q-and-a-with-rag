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
    vector_store: ChromaVectorStore,
    k: int
):
    """
    Retrieve relevant documents using similarity search.
    
    Args:
        state: Current state
        vector_store: Vector store instance
        k: Number of documents to retrieve
    """
    docs = vector_store.similarity_search(state["question"], k=k)
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
    llm: Any,
    config: Any
):
    """
    Build LangGraph for RAG pipeline.
    
    Args:
        vector_store: Vector store instance
        llm: LLM instance
        config: Configuration object
    """
    # Wrap steps to pass vector_store, llm, and config
    def retrieve_step(state: State):
        return retrieve(state, vector_store, config.retrieval.similarity_search_k)

    def generate_step(state: State):
        return generate(state, llm)

    graph_builder = StateGraph(State).add_sequence([retrieve_step, generate_step])
    graph_builder.add_edge(START, "retrieve_step")
    return graph_builder.compile()