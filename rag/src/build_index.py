"""
Script to build FAISS vector index from processed FAQ documents.
"""

import logging
import pickle
import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import get_project_root, load_json, ensure_dir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FAQIndexBuilder:
    """Builds and manages FAISS index for FAQ retrieval."""
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the index builder.
        
        Args:
            embedding_model_name: Name of the SentenceTransformer model
        """
        self.embedding_model_name = embedding_model_name
        self.embedding_model = None
        self.index = None
        self.metadata = []
        self.dimension = None
        
    def load_embedding_model(self) -> None:
        """Load the SentenceTransformer embedding model."""
        try:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            # Get embedding dimension
            test_embedding = self.embedding_model.encode("test")
            self.dimension = len(test_embedding)
            logger.info(f"Model loaded. Embedding dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            numpy array of embeddings
        """
        if not self.embedding_model:
            self.load_embedding_model()
        
        logger.info(f"Creating embeddings for {len(texts)} texts...")
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        logger.debug(f"Created embeddings shape: {embeddings.shape}")
        return embeddings
    
    def build_index(self, embeddings: np.ndarray) -> None:
        """
        Build FAISS index from embeddings.
        
        Args:
            embeddings: numpy array of embeddings
        """
        try:
            dimension = embeddings.shape[1]
            self.dimension = dimension
            
            # Create FAISS index (L2 distance)
            # Using IndexFlatL2 for simplicity (exact search)
            # For larger datasets, consider IndexIVFFlat or IndexHNSW
            logger.info(f"Building FAISS index with dimension {dimension}...")
            self.index = faiss.IndexFlatL2(dimension)
            
            # Add embeddings to index
            # FAISS expects float32
            embeddings = embeddings.astype('float32')
            self.index.add(embeddings)
            
            logger.info(f"Index built with {self.index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Failed to build index: {e}")
            raise
    
    def save_index(self, index_path: Path, metadata_path: Path) -> None:
        """
        Save FAISS index and metadata to disk.
        
        Args:
            index_path: Path to save the FAISS index
            metadata_path: Path to save the metadata
        """
        try:
            if not self.index:
                raise ValueError("No index to save. Build index first.")
            
            # Save FAISS index
            ensure_dir(index_path.parent)
            faiss.write_index(self.index, str(index_path))
            logger.info(f"Saved FAISS index to {index_path}")
            
            # Save metadata
            metadata_dict = {
                'embedding_model': self.embedding_model_name,
                'dimension': self.dimension,
                'num_vectors': self.index.ntotal,
                'metadata': self.metadata
            }
            
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata_dict, f)
            logger.info(f"Saved metadata to {metadata_path}")
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise


def build_faq_index(
    processed_docs_path: Path,
    index_path: Path,
    metadata_path: Path
) -> None:
    """
    Main function to build FAQ index from processed documents.
    
    Args:
        processed_docs_path: Path to processed_docs.json
        index_path: Path where FAISS index will be saved
        metadata_path: Path where metadata will be saved
    """
    try:
        logger.info("Starting index building process...")
        
        # Load processed documents
        logger.info(f"Loading processed documents from {processed_docs_path}")
        processed_docs = load_json(processed_docs_path)
        
        if not processed_docs:
            raise ValueError("No processed documents found")
        
        # Extract texts and metadata
        texts = [doc['text'] for doc in processed_docs]
        metadata = [
            {
                'id': doc.get('id', idx),
                'question': doc.get('question', ''),
                'answer': doc.get('answer', ''),
                'text': doc.get('text', '')
            }
            for idx, doc in enumerate(processed_docs)
        ]
        
        # Initialize builder
        builder = FAQIndexBuilder()
        builder.metadata = metadata
        
        # Create embeddings
        embeddings = builder.create_embeddings(texts)
        
        # Build index
        builder.build_index(embeddings)
        
        # Save index and metadata
        builder.save_index(index_path, metadata_path)
        
        logger.info("Index building complete")
        
    except Exception as e:
        logger.error(f"Index building failed: {e}")
        raise


if __name__ == "__main__":
    project_root = get_project_root()
    processed_docs_path = project_root / "data" / "processed_docs.json"
    index_path = project_root / "vectorstore" / "faiss.index"
    metadata_path = project_root / "vectorstore" / "metadata.pkl"
    
    build_faq_index(processed_docs_path, index_path, metadata_path)

