"""
Streams the CAP text dataset, splits opinions using RecursiveCharacterTextSplitter,
stores chunks in ChromaDB with metadata for filtering.

To run:
    # Process X cases for testing
    python pipeline/process_corpus.py --max-cases X

    # Resume after interruption
    python pipeline/process_corpus.py --resume
"""

import argparse
import itertools
import re
import time
from pathlib import Path

from datetime import datetime
from typing import Optional

import chromadb
from chromadb.config import Settings
from datasets import load_dataset
from huggingface_hub import login
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from tqdm import tqdm

from pipeline.config_loader import load_config


# Maps names to HuggingFace dataset field names
FIELD_MAP = {
    "case_id": "id",
    "text": "text",
    "jurisdiction": None,       # we will try to extract this later
    "date": None,       # extracted from text via regex
    "case_name": None,       # extracted from text via regex
    "court": None       # extracted from text via regex
}

JURISDICTION_SUBKEY = None
COURT_SUBKEY = None

# Month abbreviation regex
_MONTHS = (
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?"
)

# Date extraction regex
# Recognizes "Feb. 9, 1973" or "Feb 9, 1973" or "February 9, 1973"
_DATE_RE = re.compile(
    rf"({_MONTHS})\.?\s+(\d{{1,2}}),?\s+(\d{{4}})",
    re.IGNORECASE
)

# Words indicating a non-decision date
# These dates are skipped
_NON_DECISION_PREFIXES = re.compile(
    r"\b(heard|argued|submitted|rehearing|certiorari|see|as amended)\b",
    re.IGNORECASE
)

# Court recognition regex: "United States Court of Appeals, Ninth Circuit" etc.
_COURT_RE = re.compile(
    r"((?:United States|U\.S\.|Supreme)\s+(?:Court[^.:\n]{0,80}?"
    r"(?:Circuit|District|Bankruptcy|Claims|International\s+Trade)"
    r"|Supreme\s+Court[^.:\n]{0,60}?)|"
    r"(?:Court\s+of\s+(?:Appeals|Claims|Customs)[^.:\n]{0,60}?))",
    re.IGNORECASE
)

# Case name regex: everything before the first "No." or docket number
_CASE_NAME_RE = re.compile(
    r"^(.+?)\s*No\.?\s+[\d\-]+",
    re.DOTALL
)


def parse_header(text: str) -> dict:
    """
    Extracts court, decision date, and case name from a legal opinion header.

    The header is the first ~1,000 characters of the opinion and contains
    the parties, docket number, court name, and dates in a consistent format.

    Date selection explanation:
      - Takes the first date that appears AFTER the court name
      - Skips dates preceded by non-decision keywords (Argued, Heard, etc.)
      - If all dates are non-decision dates, falls back to the last date found
      - Returns year as int, full date string for display

    Returns dict with keys: court, year, date_str, case_name
    All values default to empty string / 0 if not found.
    """
    header = text[:1500]  # just take the beginning subset
    
    # Extract court
    court = ""
    court_match = _COURT_RE.search(header)
    court_end_pos = 0
    if court_match:
        court = court_match.group(0).strip().rstrip(".,")
        court_end_pos = court_match.end()

    # Extract date
    search_region = header[court_end_pos:] if court_end_pos else header
    year = 0
    date_str = ""

    all_dates = list(_DATE_RE.finditer(search_region))
    decision_date = None
    
    for match in all_dates:
        # Check for non-decision keywords
        preceding = search_region[max(0, match.start() - 30):match.start()]
        if _NON_DECISION_PREFIXES.search(preceding):
            continue  # skip non-decision dates
        decision_date = match
        break  # take the first non-skipped date

    # If all dates were skipped, take the last one found
    if decision_date is None and all_dates:
        decision_date = all_dates[-1]

    if decision_date:
        month_str, day_str, year_str = decision_date.groups()
        date_str = f"{month_str} {day_str}, {year_str}"
        try:
            year = int(year_str)
        except ValueError:
            year = 0
            
    # Extract case name
    case_name = ""
    name_match = _CASE_NAME_RE.match(header.strip())
    if name_match:
        raw = name_match.group(1).strip()
        case_name = " ".join(raw.split())[:200]

    return {
        "court": court,
        "year": year,
        "date_str": date_str,
        "case_name": case_name
    }


def get_chroma_collection(config: dict) -> chromadb.Collection:
    """
    Returns a ChromaDB collection in persistent local mode.

    Writes to data/chroma_db
    """
    persist_dir = config["chroma"]["persist_directory"]
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path = persist_dir,
        settings = Settings(anonymized_telemetry=False)
    )

    # get_or_create allows us to use an existing collection when resuming
    collection = client.get_or_create_collection(
        name = config["chroma"]["collection_name"],
        metadata = {"hnsw:space": "cosine"}   # cosine similarity
    )
    return collection


def process_corpus(config: dict, max_cases: int = None, resume: bool = False) -> None:
    min_text_len = config["corpus"]["min_text_length"]

    print("CORPUS PROCESSING (ChromaDB)")
    print(f"  ChromaDB directory: {config['chroma']['persist_directory']}")
    print(f"  Collection: {config['chroma']['collection_name']}")
    print(f"  Max cases: {max_cases or 'all'}")
    print(f"  Resume? {resume}")
    print()

    login(token=config["secrets"]["HF_TOKEN"], add_to_git_credential=False)
    collection = get_chroma_collection(config)

    existing_ids: set = set()
    if resume:
        existing = collection.get(include=[])
        existing_ids = set(existing["ids"])
        print(f"Resume with {len(existing_ids):,} chunks already in the collection")

    # Text splitter with legal section separators
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunking"]["chunk_size"],
        chunk_overlap=config["chunking"]["chunk_overlap"],
        separators=config["chunking"]["separators"],
    )

    # Stream in dataset from huggingface
    # Avoids loading everything into memory, important when we start scaling
    print(f"Streaming: {config['datasets']['cap_text_repo']}")
    ds = load_dataset(
        config["datasets"]["cap_text_repo"],
        split=config["datasets"]["hf_split"],
        streaming=True,
        token=config["secrets"]["HF_TOKEN"],
    )

    n_processed = n_skipped_jur = n_skipped_text = n_skipped_dup = n_errors = 0
    batch_docs: list[Document] = []
    batch_ids:  list[str]      = []
    BATCH_SIZE = 200

    start = time.time()
    pbar = tqdm(desc="Processing", unit="cases", dynamic_ncols=True)

    try:
        for record in ds:
            # Extract relevant fields
            case_id = str(record.get("id", "")).strip()
            text = str(record.get("text", "")).strip()

            meta_dict = record.get("metadata", {})
            source_url = meta_dict.get("url", "") if isinstance(meta_dict, dict) else ""

            if not case_id or not text or len(text) < min_text_len:
                n_skipped_text += 1
                pbar.update(1)
                continue

            # Parse court, year, date, and case_name from text header
            parsed = parse_header(text)
            court = parsed["court"]
            year = parsed["year"]
            date_str = parsed["date_str"]
            case_name = parsed["case_name"]

            # TODO: jurisdiction extraction
            # Jurisdiction is hard to parse out, so set to empty for now
            jurisdiction = ""
            
            # Split documents into chunks
            raw_doc = Document(
                page_content=text,
                metadata={
                    "case_id": case_id,
                    "case_name": case_name,
                    "jurisdiction": jurisdiction,
                    "year": year,
                    "court": court
                }
            )
            chunks: list[Document] = splitter.split_documents([raw_doc])

            if not chunks:
                n_skipped_text += 1
                pbar.update(1)
                continue

            # When resuming, de-duplicate chunks
            new_chunks_added = 0
            for i, chunk in enumerate(chunks):
                chunk_id = f"{case_id}__chunk{i}"
                if resume and chunk_id in existing_ids:
                    n_skipped_dup += 1
                    continue
                chunk.metadata["chunk_index"] = i
                chunk.metadata["chunk_id"] = chunk_id
                batch_docs.append(chunk)
                batch_ids.append(chunk_id)
                new_chunks_added += 1
            
            if new_chunks_added > 0:
                n_processed += 1
            elif resume:
                n_skipped_dup += 1
            
            pbar.update(1)

            # Flush to ChromaDB
            if len(batch_docs) >= BATCH_SIZE:
                _flush_batch(collection, batch_docs, batch_ids, embedding_dim=768)
                batch_docs.clear()
                batch_ids.clear()

            if max_cases and n_processed >= max_cases:
                print(f"\nReached max_cases={max_cases}. Stopping.")
                break

    except KeyboardInterrupt:
        print("\nInterrupted — flushing remaining batch...")
    finally:
        if batch_docs:
            _flush_batch(collection, batch_docs, batch_ids, embedding_dim=768)
        pbar.close()

    elapsed = time.time() - start
    total_chunks = collection.count()
    print()
    print("PROCESSING COMPLETE")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"  Cases processed: {n_processed:,}")
    print(f"  Skipped (jur): {n_skipped_jur:,}")
    print(f"  Skipped (text): {n_skipped_text:,}")
    print(f"  Skipped (dup): {n_skipped_dup:,}")
    print(f"  Errors: {n_errors:,}")
    print(f"  Total chunks in ChromaDB: {total_chunks:,}")


def _flush_batch(
    collection: chromadb.Collection,
    docs: list[Document],
    ids: list[str],
    embedding_dim: int = 768    # same dimensionality as bge embeddings
) -> None:
    """
    Adds a batch of documents to ChromaDB.
    Inserts zero-vector placeholders, with real embeddings computed and
    injected by compute_embeddings.py later.
    """
    import numpy as np
    n = len(docs)
    placeholder_embeddings = np.zeros((n, embedding_dim), dtype=np.float32).tolist()

    collection.add(
        ids=ids,
        documents=[d.page_content for d in docs],
        metadatas=[d.metadata for d in docs],
        embeddings=placeholder_embeddings
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CAP corpus into ChromaDB")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    process_corpus(cfg, max_cases=args.max_cases, resume=args.resume)