"""
Data ingestion script for loading and preprocessing FAQ dataset.
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
from datasets import load_dataset

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import get_project_root, save_json, format_qa_pair, ensure_dir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_faq_dataset() -> List[Dict[str, Any]]:
    """
    Load the FAQ dataset from Hugging Face.
    
    Returns:
        List of FAQ entries with question and answer
        
    Raises:
        Exception: If dataset loading fails
    """
    try:
        logger.info("Loading dataset from Hugging Face: MakTek/Customer_support_faqs_dataset")
        dataset = load_dataset("MakTek/Customer_support_faqs_dataset")
        
        # Extract the appropriate split (usually 'train')
        if 'train' in dataset:
            data = dataset['train']
        elif len(dataset) > 0:
            # Get the first available split
            split_name = list(dataset.keys())[0]
            data = dataset[split_name]
        else:
            raise ValueError("No data splits found in dataset")
        
        logger.info(f"Dataset loaded successfully. Total entries: {len(data)}")
        return data
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


def preprocess_faqs(data: Any) -> List[Dict[str, str]]:
    """
    Preprocess FAQ data into standardized format.
    
    Args:
        data: Dataset object from Hugging Face
        
    Returns:
        List of dictionaries with 'question', 'answer', and 'text' fields
    """
    processed_docs = []
    
    for idx, item in enumerate(data):
        try:
            # Handle different possible column names
            question = None
            answer = None
            
            # Common column name variations
            if 'question' in item and 'answer' in item:
                question = str(item['question']).strip()
                answer = str(item['answer']).strip()
            elif 'Question' in item and 'Answer' in item:
                question = str(item['Question']).strip()
                answer = str(item['Answer']).strip()
            elif 'Q' in item and 'A' in item:
                question = str(item['Q']).strip()
                answer = str(item['A']).strip()
            elif 'text' in item:
                # If dataset has combined text, try to split it
                text = str(item['text']).strip()
                if 'Q:' in text and 'A:' in text:
                    parts = text.split('A:', 1)
                    question = parts[0].replace('Q:', '').strip()
                    answer = parts[1].strip() if len(parts) > 1 else ""
                else:
                    question = text
                    answer = ""
            else:
                logger.warning(f"Unknown format at index {idx}: {list(item.keys())}")
                continue
            
            if question and answer:
                formatted_text = format_qa_pair(question, answer)
                processed_docs.append({
                    'id': idx,
                    'question': question,
                    'answer': answer,
                    'text': formatted_text
                })
            else:
                logger.warning(f"Skipping entry {idx}: missing question or answer")
                
        except Exception as e:
            logger.warning(f"Error processing entry {idx}: {e}")
            continue
    
    logger.info(f"Processed {len(processed_docs)} FAQ entries")
    return processed_docs


def ingest_dataset(output_path: Path) -> None:
    """
    Main ingestion function: load, preprocess, and save FAQ data.
    
    Args:
        output_path: Path where processed data will be saved
    """
    try:
        logger.info("Starting data ingestion process...")
        
        # Load dataset
        raw_data = load_faq_dataset()
        
        # Preprocess
        processed_docs = preprocess_faqs(raw_data)
        
        if not processed_docs:
            raise ValueError("No valid FAQ entries processed")
        
        # Save processed data
        save_json(processed_docs, output_path)
        
        logger.info(f"Data ingestion complete. Saved {len(processed_docs)} entries to {output_path}")
        
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        raise


if __name__ == "__main__":
    project_root = get_project_root()
    output_file = project_root / "data" / "processed_docs.json"
    
    ingest_dataset(output_file)

