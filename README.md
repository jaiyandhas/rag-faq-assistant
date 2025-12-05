# RAG FAQ Assistant

A Retrieval-Augmented Generation (RAG) system for answering customer support FAQs using LangChain, FAISS, and Streamlit.

## Overview

This project implements a RAG-based FAQ assistant that:
- Loads FAQ datasets from Hugging Face
- Creates embeddings using SentenceTransformers
- Builds a FAISS vector index for fast similarity search
- Retrieves relevant context and generates answers using LLMs
- Provides an interactive Streamlit chat interface

## Architecture

```
User Query
    ↓
[Embedding Model] → Query Vector
    ↓
[FAISS Index] → Retrieve Top-K Documents
    ↓
[Context + Query] → [LLM] → Generated Answer
    ↓
Streamlit UI
```

### Components

1. **Data Ingestion** (`src/ingest.py`): Loads and preprocesses FAQ dataset
2. **Index Building** (`src/build_index.py`): Creates embeddings and FAISS index
3. **RAG Pipeline** (`src/rag_pipeline.py`): Retrieval and generation logic
4. **Streamlit UI** (`app.py`): Interactive chat interface

## Tech Stack

- **Python** 3.8+
- **LangChain**: LLM orchestration
- **FAISS**: Vector similarity search
- **SentenceTransformers**: Text embeddings (all-MiniLM-L6-v2)
- **Streamlit**: Web UI
- **OpenAI GPT-4o mini**: Primary LLM (with fallback support for local models)
- **Hugging Face Datasets**: Data loading

## Project Structure

```
rag-faq-bot/
├── src/
│   ├── ingest.py          # Data ingestion script
│   ├── build_index.py     # FAISS index builder
│   ├── rag_pipeline.py    # RAG pipeline implementation
│   └── utils.py           # Utility functions
├── data/
│   └── processed_docs.json # Processed FAQ data
├── models/                 # (Optional) Local model storage
├── vectorstore/
│   ├── faiss.index        # FAISS vector index
│   └── metadata.pkl       # Document metadata
├── app.py                  # Streamlit UI
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md
```

## Setup Instructions

### 1. Clone and Install

```bash
# Navigate to project directory
cd rag-faq-bot

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

**Note**: OpenAI API key is optional. Without it, the system will use a fallback mode (returns best matching FAQ answer).

### 3. Run Data Pipeline

```bash
# Step 1: Ingest dataset
python src/ingest.py

# Step 2: Build FAISS index
python src/build_index.py
```

### 4. Launch Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### 5. Verify Setup (Optional)

Run the verification script to check your setup:

```bash
python verify_setup.py
```

## Usage

### Command Line Scripts

| Script | Purpose |
|--------|---------|
| `python src/ingest.py` | Loads FAQ dataset from Hugging Face and preprocesses |
| `python src/build_index.py` | Creates embeddings and builds FAISS index |
| `python verify_setup.py` | Verifies setup and checks dependencies |
| `streamlit run app.py` | Launches the Streamlit chat interface |

### Streamlit UI Features

- **Chat Interface**: Ask questions in natural language
- **Retrieved Context**: View the FAQ documents used for answering
- **Debug Mode**: Toggle to see similarity scores and retrieved chunks
- **Configurable Retrieval**: Adjust number of documents retrieved (1-10)

## Configuration

### Embedding Model

Default: `all-MiniLM-L6-v2` (can be changed in `src/build_index.py`)

### LLM Model

Default: `gpt-4o-mini` (can be changed in `src/rag_pipeline.py`)

### Retrieval Parameters

- **k (number of documents)**: Default 3, adjustable in Streamlit sidebar

## Screenshots

Add screenshots of the application int<img width="1470" height="956" alt="Screenshot 2025-11-04 at 8 02 32 AM" src="https://github.com/user-attachments/assets/ef6de9c9-a417-4e35-8ea5-524bf4387870" />
<img width="1470" height="956" alt="Screenshot 2025-11-04 at 8 12 55 AM" src="https://github.com/user-attachments/assets/aae9ee11-a3b6-4d46-8a30-26fc9d65b320" />
[RAG FAQ Assistant.pdf](https://github.com/user-attachments/files/23322384/RAG.FAQ.Assistant.pdf)


## Testing

To test the pipeline components individually:

```python
# Test ingestion
from src.ingest import ingest_dataset
from pathlib import Path
ingest_dataset(Path("data/processed_docs.json"))

# Test RAG pipeline
from src.rag_pipeline import load_rag_pipeline
pipeline = load_rag_pipeline()
result = pipeline.query("What is your return policy?")
print(result['answer'])
```

## Troubleshooting

### Issue: "Index files not found"
**Solution**: Run `python src/build_index.py` first

### Issue: "OPENAI_API_KEY not found"
**Solution**: 
- Create `.env` file with `OPENAI_API_KEY=your_key`
- Or the system will use fallback mode (best matching FAQ)

### Issue: "Dataset loading fails"
**Solution**: Check internet connection and Hugging Face dataset availability

### Issue: "FAISS index build is slow"
**Solution**: For large datasets, consider using `IndexIVFFlat` or `IndexHNSW` instead of `IndexFlatL2`

## Development

### Adding Custom Datasets

Modify `src/ingest.py` to load your own dataset:

```python
# Replace load_dataset call with your dataset
dataset = load_dataset("your_dataset_name")
```

### Customizing the System Prompt

Edit the `system_prompt` in `src/rag_pipeline.py`:

```python
self.system_prompt = "Your custom prompt here..."
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Dataset: [MakTek/Customer_support_faqs_dataset](https://huggingface.co/datasets/MakTek/Customer_support_faqs_dataset)
- SentenceTransformers by UKP Lab
- LangChain framework
- FAISS by Facebook Research

## Contact

For questions or issues, please open an issue on GitHub.

