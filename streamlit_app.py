import os
import streamlit as st
from src.loader import load_documents
from src.splitter import split_documents
from src.vectordb import get_vector_db
from src.retriever import retrieve_documents
from src.prompt import format_context, build_prompt
from src.llm import generate_answer

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 Institutional PDF Assistant")
st.write("Ask questions and chat naturally with your college documents.")

@st.cache_resource
def initialize_rag():
    """Load docs and initialize vector database once."""
    documents = load_documents()
    chunks = split_documents(documents)
    vector_db = get_vector_db(chunks)
    return vector_db

# Initialize database
with st.spinner("Loading Vector Database..."):
    vector_db = initialize_rag()

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages from history when page re-runs
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("View Sources"):
                for source in message["sources"]:
                    st.markdown(f"- {source}")

# Accept user input from the bottom chat box (just like ChatGPT)
if prompt_query := st.chat_input("Ask a question about your documents..."):
    
    # 1. Add user message to session state and display it
    st.session_state.messages.append({"role": "user", "content": prompt_query})
    with st.chat_message("user"):
        st.markdown(prompt_query)

    # 2. Generate assistant response using RAG
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Retrieve relevant chunks
            results = retrieve_documents(vector_db, prompt_query)
            
            if not results:
                answer = "I don't have enough information in the provided documents to answer that."
                sources = []
            else:
                # Build context & prompt
                context = format_context(results)
                prompt = build_prompt(context, prompt_query)
                
                # Generate answer via Groq
                answer = generate_answer(prompt)
                
                # Extract sources
                sources = set()
                for doc in results:
                    source_path = doc.metadata.get("source", "Unknown File")
                    file_name = os.path.basename(source_path)
                    page_num = doc.metadata.get("page", 0) + 1
                    sources.add(f"{file_name} (Page {page_num})")
                sources = list(sources)

            # Display the answer
            st.markdown(answer)
            
            # Display sources neatly inside an expandable dropdown
            if sources:
                with st.expander("View Sources"):
                    for source in sources:
                        st.markdown(f"- {source}")

            # 3. Add assistant response to session state history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer, 
                "sources": sources
            })