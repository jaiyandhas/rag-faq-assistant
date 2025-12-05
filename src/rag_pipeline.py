"""
RAG Pipeline for FAQ retrieval and generation.
"""

import logging
import pickle
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# LangChain imports - handle both old and new API versions
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:
    try:
        from langchain.chat_models import ChatOpenAI
        from langchain.schema import HumanMessage, SystemMessage
    except ImportError:
        # Fallback if LangChain is not available
        ChatOpenAI = None
        HumanMessage = None
        SystemMessage = None

from src.utils import get_project_root

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline for FAQ answering."""
    
    def __init__(
        self,
        index_path: Path,
        metadata_path: Path,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        llm_model: str = "gpt-4o-mini",
        use_openai: bool = True,
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            index_path: Path to FAISS index
            metadata_path: Path to metadata pickle file
            embedding_model_name: Name of embedding model
            llm_model: Name of LLM model
            use_openai: Whether to use OpenAI (True) or fallback to local model
            openai_api_key: OpenAI API key (if None, will try to get from env)
        """
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.embedding_model_name = embedding_model_name
        self.llm_model = llm_model
        self.use_openai = use_openai
        
        # Load components
        self.index = None
        self.metadata = None
        self.embedding_model = None
        self.llm = None
        
        # System prompt for LLM
        self.system_prompt = """You are a helpful customer support assistant. 
Answer questions based ONLY on the provided context. 
If the answer cannot be found in the context, say: "I don't know based on my current data."
Be concise, accurate, and helpful. Use the exact information from the context when available."""
        
        self._load_components(openai_api_key)
    
    def _load_components(self, openai_api_key: Optional[str] = None) -> None:
        """Load all required components (index, embeddings, LLM)."""
        try:
            # Load FAISS index
            logger.info(f"Loading FAISS index from {self.index_path}")
            self.index = faiss.read_index(str(self.index_path))
            logger.info(f"Loaded index with {self.index.ntotal} vectors")
            
            # Load metadata
            logger.debug(f"Loading metadata from {self.metadata_path}")
            with open(self.metadata_path, 'rb') as f:
                metadata_dict = pickle.load(f)
                self.metadata = metadata_dict['metadata']
            logger.info(f"Loaded metadata for {len(self.metadata)} documents")
            
            # Load embedding model
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.debug("Embedding model loaded")
            
            # Initialize LLM
            if self.use_openai:
                import os
                api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
                if not api_key:
                    logger.warning("OPENAI_API_KEY not found. Using fallback mode.")
                    self.use_openai = False
                    self._init_local_llm()
                else:
                    if ChatOpenAI is None:
                        logger.warning("LangChain not available. Using fallback mode.")
                        self.use_openai = False
                        self._init_local_llm()
                    else:
                        logger.info(f"Initializing OpenAI LLM: {self.llm_model}")
                        try:
                            self.llm = ChatOpenAI(
                                model=self.llm_model,
                                temperature=0.0,
                                api_key=api_key
                            )
                        except TypeError:
                            self.llm = ChatOpenAI(
                                model_name=self.llm_model,
                                temperature=0.0,
                                openai_api_key=api_key
                            )
                        logger.debug("OpenAI LLM initialized")
            else:
                self._init_local_llm()
                
        except Exception as e:
            logger.error(f"Failed to load components: {e}")
            raise
    
    def _init_local_llm(self) -> None:
        """Initialize local LLM (Llama/Mistral) as fallback."""
        logger.debug("Local LLM not available, using fallback mode")
        self.llm = None
    
    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant documents for a query.
        
        Args:
            query: User query string
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved document metadata
        """
        try:
            # Create query embedding
            query_embedding = self.embedding_model.encode([query])
            query_embedding = query_embedding.astype('float32')
            
            # Search in FAISS index
            k = min(k, self.index.ntotal)  # Ensure k doesn't exceed index size
            distances, indices = self.index.search(query_embedding, k)
            
            # Retrieve metadata for top-k results
            retrieved_docs = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.metadata):
                    doc = self.metadata[idx].copy()
                    doc['similarity_score'] = float(1 / (1 + distance))  # Convert distance to similarity
                    retrieved_docs.append(doc)
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents for query")
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []
    
    def generate_answer(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        Generate answer using LLM with retrieved context.
        
        Args:
            query: User query
            context_docs: Retrieved context documents
            
        Returns:
            Generated answer string
        """
        try:
            # Combine context
            context_text = "\n\n".join([doc['text'] for doc in context_docs])
            
            # Format prompt
            if self.llm and self.use_openai and SystemMessage and HumanMessage:
                messages = [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(content=f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer:")
                ]
                try:
                    response = self.llm.invoke(messages)
                    # Handle both string and object responses
                    if hasattr(response, 'content'):
                        return response.content
                    elif isinstance(response, str):
                        return response
                    else:
                        return str(response)
                except Exception as invoke_error:
                    logger.error(f"Error during LLM invocation: {invoke_error}", exc_info=True)
                    # Fallback to best match
                    if context_docs:
                        best_match = context_docs[0]
                        logger.debug("Using fallback: returning best match answer")
                        return best_match['answer']
                    else:
                        return "I don't know based on my current data."
            else:
                if context_docs:
                    best_match = context_docs[0]
                    return best_match['answer']
                else:
                    return "I don't know based on my current data."
                    
        except Exception as e:
            logger.error(f"Error during generation: {e}", exc_info=True)
            if context_docs:
                best_match = context_docs[0]
                return best_match['answer']
            return "I encountered an error while generating an answer. Please try again."
    
    def query(self, question: str, k: int = 3) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve and generate.
        
        Args:
            question: User question
            k: Number of documents to retrieve
            
        Returns:
            Dictionary with answer, retrieved_docs, and metadata
        """
        try:
            # Retrieve
            retrieved_docs = self.retrieve(question, k=k)
            
            # Generate
            answer = self.generate_answer(question, retrieved_docs)
            
            return {
                'question': question,
                'answer': answer,
                'retrieved_docs': retrieved_docs,
                'num_retrieved': len(retrieved_docs)
            }
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            return {
                'question': question,
                'answer': "I encountered an error. Please try again.",
                'retrieved_docs': [],
                'num_retrieved': 0,
                'error': str(e)
            }


def load_rag_pipeline(
    index_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
    use_openai: bool = True,
    openai_api_key: Optional[str] = None
) -> RAGPipeline:
    """
    Convenience function to load RAG pipeline with default paths.
    
    Args:
        index_path: Optional path to index (defaults to project vectorstore)
        metadata_path: Optional path to metadata (defaults to project vectorstore)
        use_openai: Whether to use OpenAI
        openai_api_key: Optional OpenAI API key
        
    Returns:
        Initialized RAGPipeline instance
    """
    if index_path is None or metadata_path is None:
        project_root = get_project_root()
        if index_path is None:
            index_path = project_root / "vectorstore" / "faiss.index"
        if metadata_path is None:
            metadata_path = project_root / "vectorstore" / "metadata.pkl"
    
    return RAGPipeline(
        index_path=index_path,
        metadata_path=metadata_path,
        use_openai=use_openai,
        openai_api_key=openai_api_key
    )

