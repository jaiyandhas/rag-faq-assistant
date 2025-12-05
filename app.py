"""
Streamlit UI for RAG FAQ Assistant.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.rag_pipeline import load_rag_pipeline
from src.utils import get_project_root

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="RAG FAQ Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'initialized' not in st.session_state:
    st.session_state.initialized = False


@st.cache_resource
def initialize_rag_pipeline() -> Optional[object]:
    """Initialize and cache RAG pipeline."""
    try:
        project_root = get_project_root()
        index_path = project_root / "vectorstore" / "faiss.index"
        metadata_path = project_root / "vectorstore" / "metadata.pkl"
        
        # Check if files exist
        if not index_path.exists() or not metadata_path.exists():
            return None
        
        # Get OpenAI API key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        use_openai = st.sidebar.checkbox("Use OpenAI (requires API key)", value=bool(openai_api_key))
        
        pipeline = load_rag_pipeline(
            index_path=index_path,
            metadata_path=metadata_path,
            use_openai=use_openai,
            openai_api_key=openai_api_key
        )
        
        return pipeline
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        return None


def main():
    """Main Streamlit app."""
    
    st.title("RAG FAQ Assistant")
    st.markdown("Ask questions about customer support FAQs")
    
    with st.sidebar:
        st.header("Documentation")
        st.markdown("""
        ### How to Use
        1. Type your question in the chat input
        2. The system will retrieve relevant FAQs
        3. An answer will appear
        
        ### Setup Instructions
        1. Run `python src/ingest.py` to load data
        2. Run `python src/build_index.py` to build index
        3. Set `OPENAI_API_KEY` environment variable (optional)
        4. Start this app with `streamlit run app.py`
        """)
        
        st.header("Configuration")
        show_debug = st.checkbox("Show Debug Logs", value=False)
        num_retrievals = st.slider("Number of documents to retrieve", 1, 10, 3)
        
        st.header("Status")
        
        # Check if index exists
        project_root = get_project_root()
        index_path = project_root / "vectorstore" / "faiss.index"
        metadata_path = project_root / "vectorstore" / "metadata.pkl"
        
        if index_path.exists() and metadata_path.exists():
            st.success("Index files found")
            
            if st.session_state.rag_pipeline is None:
                with st.spinner("Loading RAG pipeline..."):
                    st.session_state.rag_pipeline = initialize_rag_pipeline()
            
            if st.session_state.rag_pipeline:
                st.success("RAG Pipeline ready")
            else:
                st.error("Failed to load RAG pipeline")
        else:
            st.error("Index files not found")
            st.info("Run `python src/build_index.py` first")
    
    if st.session_state.rag_pipeline is None:
        st.warning("RAG pipeline not initialized. Please check the sidebar for setup instructions.")
        st.stop()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant" and "retrieved_docs" in message:
                with st.expander("Retrieved Context"):
                    for i, doc in enumerate(message["retrieved_docs"][:num_retrievals], 1):
                        st.markdown(f"**Document {i}** (similarity: {doc.get('similarity_score', 0):.3f})")
                        st.markdown(f"**Q:** {doc.get('question', 'N/A')}")
                        st.markdown(f"**A:** {doc.get('answer', 'N/A')}")
                        st.divider()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about customer support..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.rag_pipeline.query(
                        question=prompt,
                        k=num_retrievals
                    )
                    
                    answer = result['answer']
                    retrieved_docs = result.get('retrieved_docs', [])
                    
                    # Display answer
                    st.markdown(answer)
                    
                    if show_debug and retrieved_docs:
                        with st.expander("Debug: Retrieved Documents"):
                            for i, doc in enumerate(retrieved_docs, 1):
                                st.markdown(f"**Doc {i}** (score: {doc.get('similarity_score', 0):.4f})")
                                st.code(doc.get('text', ''), language=None)
                    
                    # Store in session
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "retrieved_docs": retrieved_docs
                    })
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    logger.error(f"Error in chat: {e}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })


if __name__ == "__main__":
    main()

