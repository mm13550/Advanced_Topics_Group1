"""
Downloads the ChromaDB index from HuggingFace and extracts it to data/chroma_db/

Requires "pip install huggingface_hub"
No HuggingFace account needed as dataset is public

The current database contains 10,000 federal circuit court opinions from the
Caselaw Access Project, chunked and embedded with BAAI/bge-base-en-v1.5.

Total size: ~1.6 GB compressed.
"""

import os
import tarfile
from pathlib import Path


HF_REPO_ID  = "hanwenzhang01/legal-rag-chromadb-10k"
FILENAME    = "chroma_db_10k.tar.gz"
EXTRACT_DIR = "data"
CHROMA_DIR  = "data/chroma_db"


def download_db():
    from huggingface_hub import hf_hub_download

    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        print(f"ChromaDB already exists at {CHROMA_DIR}/")
        print("Delete data/chroma_db/ and rerun to force a fresh download.")
        return

    Path(EXTRACT_DIR).mkdir(parents=True, exist_ok=True)

    print(f"Downloading ChromaDB index from HuggingFace...")
    print(f"  Repository: {HF_REPO_ID}")
    print(f"  File: {FILENAME}")
    print()

    local_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=FILENAME,
        repo_type="dataset",
        local_dir=EXTRACT_DIR
    )

    print(f"Downloaded to: {local_path}")
    print("Extracting...")

    with tarfile.open(local_path, "r:gz") as tar:
        tar.extractall(path=EXTRACT_DIR)

    print(f"Extracted to: {CHROMA_DIR}/")

    # Clean up compressed file
    os.remove(local_path)
    print("Removed compressed file.")

    print()
    print("Database ready. Verify with:")
    print("  python -m pipeline.smoke_test --week 2")


if __name__ == "__main__":
    download_db()