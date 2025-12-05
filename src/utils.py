"""
Utility functions for the RAG FAQ Assistant.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def ensure_dir(directory: Path) -> None:
    """Ensure a directory exists, create if it doesn't."""
    directory.mkdir(parents=True, exist_ok=True)


def load_json(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load JSON data from a file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of dictionaries loaded from JSON
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.debug(f"Loaded JSON from {file_path}")
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path}: {e}")
        raise


def save_json(data: List[Dict[str, Any]], file_path: Path) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: List of dictionaries to save
        file_path: Path where to save the JSON file
    """
    try:
        ensure_dir(file_path.parent)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.debug(f"Saved JSON to {file_path}")
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {e}")
        raise


def format_qa_pair(question: str, answer: str) -> str:
    """
    Format a Q/A pair into a standardized string.
    
    Args:
        question: The question text
        answer: The answer text
        
    Returns:
        Formatted string: "Q: ...\nA: ..."
    """
    return f"Q: {question}\nA: {answer}"

