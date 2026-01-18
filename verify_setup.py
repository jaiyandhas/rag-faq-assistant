"""
Verification script to check if the RAG FAQ Assistant is properly set up.
"""

import sys
from pathlib import Path

def check_file_exists(filepath: Path, description: str) -> bool:
    """Check if a file exists and print status."""
    exists = filepath.exists()
    status = "[OK]" if exists else "[MISSING]"
    print(f"{status} {description}: {filepath}")
    return exists

def check_imports() -> bool:
    """Check if required packages can be imported."""
    print("\nChecking Python packages...")
    packages = [
        ("datasets", "Hugging Face datasets"),
        ("sentence_transformers", "SentenceTransformers"),
        ("faiss", "FAISS"),
        ("streamlit", "Streamlit"),
        ("langchain", "LangChain"),
    ]
    
    all_ok = True
    for package_name, description in packages:
        try:
            __import__(package_name)
            print(f"[OK] {description} ({package_name})")
        except ImportError:
            print(f"[MISSING] {description} ({package_name}) - NOT INSTALLED")
            all_ok = False
    
    return all_ok

def main():
    """Run setup verification."""
    print("RAG FAQ Assistant - Setup Verification\n")
    print("=" * 50)
    
    project_root = Path(__file__).parent
    
    print("\nChecking directory structure...")
    dirs_ok = True
    dirs_ok &= check_file_exists(project_root / "src", "src/ directory")
    dirs_ok &= check_file_exists(project_root / "data", "data/ directory")
    dirs_ok &= check_file_exists(project_root / "vectorstore", "vectorstore/ directory")
    
    print("\nChecking key files...")
    files_ok = True
    files_ok &= check_file_exists(project_root / "app.py", "Streamlit app")
    files_ok &= check_file_exists(project_root / "requirements.txt", "requirements.txt")
    files_ok &= check_file_exists(project_root / "README.md", "README.md")
    files_ok &= check_file_exists(project_root / "src" / "ingest.py", "ingest.py")
    files_ok &= check_file_exists(project_root / "src" / "build_index.py", "build_index.py")
    files_ok &= check_file_exists(project_root / "src" / "rag_pipeline.py", "rag_pipeline.py")
    
    print("\nChecking processed data...")
    data_ok = check_file_exists(
        project_root / "data" / "processed_docs.json",
        "Processed FAQ data"
    )
    
    print("\nChecking vector index...")
    index_ok = check_file_exists(
        project_root / "vectorstore" / "faiss.index",
        "FAISS index"
    )
    metadata_ok = check_file_exists(
        project_root / "vectorstore" / "metadata.pkl",
        "Index metadata"
    )
    
    packages_ok = check_imports()
    
    print("\nChecking environment...")
    import os
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print("[OK] OPENAI_API_KEY is set")
    else:
        print("[INFO] OPENAI_API_KEY not set (optional - fallback mode will be used)")
    
    print("\n" + "=" * 50)
    print("\nSummary:")
    
    if not dirs_ok or not files_ok:
        print("[ERROR] Project structure is incomplete. Please check the missing files.")
        sys.exit(1)
    
    if not packages_ok:
        print("[ERROR] Some required packages are missing. Run: pip install -r requirements.txt")
        sys.exit(1)
    
    if not data_ok:
        print("[WARNING] Processed data not found. Run: python src/ingest.py")
    
    if not index_ok or not metadata_ok:
        print("[WARNING] Vector index not found. Run: python src/build_index.py")
    
    if data_ok and index_ok and metadata_ok and packages_ok:
        print("[OK] All checks passed! You're ready to run the app.")
        print("\nStart the app with: streamlit run app.py")
    else:
        print("[WARNING] Setup incomplete. Follow the warnings above to complete setup.")

if __name__ == "__main__":
    main()

