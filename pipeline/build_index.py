"""
Loads precomputed BGE embeddings from HuggingFace and injects them
into the existing ChromaDB collection built by process_corpus.py
"""

import argparse
import itertools
import time
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings
from datasets import load_dataset
from huggingface_hub import login
from tqdm import tqdm

from pipeline.config_loader import load_config
from pipeline.process_corpus import get_chroma_collection


def build_index(config: dict) -> None:
    print("BUILD EMBEDDING INDEX")

    login(token=config["secrets"]["HF_TOKEN"], add_to_git_credential=False)

    # Load ChromaDB collection
    collection = get_chroma_collection(config)
    total_chunks = collection.count()
    print(f"ChromaDB chunks already stored: {total_chunks:,}")

    # Get all case_ids with chunk_index=0
    # These are the chunks with precomputed embeddings
    print("  Fetching primary chunk IDs (chunk_index=0)...")
    primary = collection.get(
        where={"chunk_index": 0},
        include=["metadatas"]
    )
    primary_case_ids = {
        meta["case_id"]: chunk_id
        for chunk_id, meta in zip(primary["ids"], primary["metadatas"])
    }
    print(f"Primary chunks: {len(primary_case_ids):,}")

    # Stream precomputed embeddings dataset of768-dim BGE vectors, pre-normalized
    print(f"\nStreaming: {config['datasets']['cap_embeddings_repo']}")

    emb_ds = load_dataset(
        config["datasets"]["cap_embeddings_repo"],
        split=config["datasets"]["hf_split"],
        streaming=True,
        token=config["secrets"]["HF_TOKEN"]
    )

    # Inject embeddings in batches
    BATCH_SIZE = 500
    batch_ids: list[str] = []
    batch_embs: list[list[float]] = []

    matched = skipped = 0
    start = time.time()
    pbar = tqdm(desc="Injecting embeddings", unit="cases", dynamic_ncols=True)

    try:
        for record in emb_ds:
            # Confirm actual field name from inspect_schema.py output
            case_id = str(record.get("id", record.get("case_id", "")))
            if not case_id or case_id not in primary_case_ids:
                skipped += 1
                pbar.update(1)
                continue

            embedding = record.get("embedding", [])
            if not embedding:
                skipped += 1
                pbar.update(1)
                continue

            # Ensure normalization
            vec = np.array(embedding, dtype="float32")
            norm = np.linalg.norm(vec)
            if norm > 0 and abs(norm - 1.0) > 0.01:
                vec = vec / norm

            chunk_id = primary_case_ids[case_id]
            batch_ids.append(chunk_id)
            batch_embs.append(vec.tolist())
            matched += 1
            pbar.update(1)

            if len(batch_ids) >= BATCH_SIZE:
                collection.update(ids=batch_ids, embeddings=batch_embs)
                batch_ids.clear()
                batch_embs.clear()

    except KeyboardInterrupt:
        print("\nInterrupted — flushing batch...")
    finally:
        if batch_ids:
            collection.update(ids=batch_ids, embeddings=batch_embs)
        pbar.close()

    elapsed = time.time() - start
    print()
    print("INDEX BUILD COMPLETE")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"  Embeddings matched and injected: {matched:,}")
    print(f"  Cases skipped (not in corpus): {skipped:,}")
    print(f"  Coverage: {100*matched/len(primary_case_ids):.1f}% of primary chunks have embeddings")
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject precomputed embeddings into ChromaDB")
    parser.add_argument("--config", default="configs/config.yaml")

    args = parser.parse_args()

    cfg = load_config(args.config)
    build_index(cfg, dry_run=args.dry_run)