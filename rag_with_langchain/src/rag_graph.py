from langchain import hub
from langchain.schema import Document
from langchain_community.chat_models import ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import StateGraph, START
from typing import List, TypedDict

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
def retrieve(state: State, vector_store: InMemoryVectorStore):
    docs = vector_store.similarity_search(state["question"], k=3)
    return {"context": docs}

def generate(state: State, llm: ChatOpenAI):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

def build_langgraph(vector_store, llm: ChatOpenAI):
    # Wrap steps to pass vector_store and llm
    def retrieve_step(state: State):
        return retrieve(state, vector_store)

    def generate_step(state: State):
        return generate(state, llm)

    graph_builder = StateGraph(State).add_sequence([retrieve_step, generate_step])
    graph_builder.add_edge(START, "retrieve_step")
    return graph_builder.compile()