import streamlit as st
import asyncio
import wikipedia
import os
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext
)
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import FunctionCallingAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.response_synthesizers import StreamingResponseSynthesizer

# Create directory for uploaded files
UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ✅ Define custom tools
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the result."""
    return a * b

def wikipedia_search(query: str, sentences: int = 2) -> str:
    """
    Search Wikipedia for a query and return the summary.
    """
    try:
        summary = wikipedia.summary(query, sentences=sentences)
        return summary
    except Exception as e:
        return f"Error fetching Wikipedia article: {e}"

# ✅ Function to load and index uploaded files
def load_documents():
    documents = SimpleDirectoryReader(UPLOAD_DIR).load_data()
    return documents

# ✅ Cache agent setup
@st.cache_resource(show_spinner=False)
def setup_rag_agent(documents):
    # Build index from documents
    index = VectorStoreIndex.from_documents(documents)
    retriever = index.as_retriever(similarity_top_k=3)

    # Wrap functions as tools
    multiply_tool = FunctionTool.from_defaults(fn=multiply)
    wiki_tool = FunctionTool.from_defaults(fn=wikipedia_search)

    # Initialize OpenAI LLM with system prompt
    llm = OpenAI(
        model="gpt-4o",
        system_prompt=(
            "You are an intelligent assistant. "
            "Answer questions using the retrieved documents when possible. "
            "Use tools for calculations or Wikipedia lookups when required. "
            "Be concise and factual, and never make up information."
        ),
        streaming=True  # ✅ Enable streaming
    )

    # Streaming response synthesizer
    response_synthesizer = StreamingResponseSynthesizer(
        service_context=ServiceContext.from_defaults(llm=llm)
    )

    # Create agent
    agent = FunctionCallingAgent.from_tools(
        tools=[multiply_tool, wiki_tool],
        retriever=retriever,
        llm=llm,
        response_synthesizer=response_synthesizer
    )
    return agent

# ✅ Streamlit UI
st.set_page_config(page_title="🦙 RAG Chatbot", page_icon="🤖")
st.title("🦙 RAG + Tools Chatbot")
st.markdown("Upload PDFs or Markdown files, then ask me anything!")

# 📤 File uploader
uploaded_files = st.file_uploader(
    "Upload your documents (PDF, Markdown, or Text):",
    type=["pdf", "md", "txt"],
    accept_multiple_files=True
)

# Save uploaded files
if uploaded_files:
    for file in uploaded_files:
        file_path = os.path.join(UPLOAD_DIR, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
    st.success(f"Uploaded {len(uploaded_files)} file(s).")

# Load documents and setup agent
documents = load_documents()
if documents:
    agent = setup_rag_agent(documents)
else:
    st.warning("📂 Please upload at least one document to enable RAG.")

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 📝 Chat input
if documents:
    user_input = st.text_input("Your question:", "")

    # Handle user input
    if user_input:
        with st.spinner("Thinking..."):
            # Save user message
            st.session_state.chat_history.append(("🧑 You", user_input))
            # Stream response
            response_container = st.empty()
            full_response = ""

            async def stream_response():
                async for token in agent.astream(user_input):
                    nonlocal full_response
                    full_response += token
                    response_container.markdown(f"**🤖 Assistant:** {full_response}")

            asyncio.run(stream_response())

            # Save assistant response
            st.session_state.chat_history.append(("🤖 Assistant", full_response))

# 💬 Display chat history
for sender, message in st.session_state.chat_history:
    st.markdown(f"**{sender}:** {message}")
