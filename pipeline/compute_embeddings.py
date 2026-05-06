"""
Computes BGE embeddings for all chunks stored in ChromaDB and
updates the collection with the computed vectors.

Runs after process_corpus.py. Replaces build_index.py for
self-computed embeddings.

To run:
    python -m pipeline.compute_embeddings

    # With explicit batch size
    python -m pipeline.compute_embeddings --batch-size 256
"""

import argparse
import time
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from pipeline.config_loader import load_config


def compute_embeddings(config: dict, batch_size: int = 512) -> None:
    persist_dir = config["chroma"]["persist_directory"]
    collection_name = config["chroma"]["collection_name"]
    embedding_model = config["embeddings"]["model"]

    print("COMPUTING CHUNK EMBEDDINGS")
    print(f"  ChromaDB directory: {persist_dir}")
    print(f"  Model: {embedding_model}")
    print(f"  Batch size: {batch_size}")
    print()

    # Load embedding model
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = SentenceTransformer(embedding_model, device=device)
    print(f"Model loaded on {device}")

    # Connect to ChromaDB
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_collection(collection_name)
    total = collection.count()
    print(f"Total chunks in collection: {total:,}")

    # Get all chunks that need embeddings
    all_ids_result = collection.get(include=[])
    all_ids = all_ids_result["ids"]
    print(f"{len(all_ids):,} chunks to embed")

    # Filter out already-embedded chunks
    print("Checking for existing embeddings...")
    already_embedded = []
    for i in range(0, len(all_ids), batch_size):
        batch_ids = all_ids[i : i + batch_size]
        batch_result = collection.get(
            ids=batch_ids,
            include=["embeddings"]
        )
        for uid, emb in zip(batch_ids, batch_result["embeddings"]):
            if emb is not None and np.any(np.array(emb) != 0.0):
                already_embedded.append(uid)

    all_ids = [uid for uid in all_ids if uid not in set(already_embedded)]
    print(f"  {len(already_embedded):,} already embedded, skipping")
    print(f"  {len(all_ids):,} chunks to embed")

    # Embed in batches
    start = time.time()
    n_processed = 0
    t_read = t_encode = t_write = 0.0

    pbar = tqdm(total=len(all_ids), desc="Embedding", unit="chunks")

    for i in range(0, len(all_ids), batch_size):
        batch_ids = all_ids[i : i + batch_size]

        # Get the text for this batch
        t0 = time.time()
        batch_result = collection.get(
            ids = batch_ids,
            include = ["documents"]
        )
        texts = batch_result["documents"]
        t_read += time.time() - t0

        # Compute embeddings
        t0 = time.time()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            device=device
        )
        t_encode += time.time() - t0

        # Update ChromaDB with computed embeddings
        t0 = time.time()
        collection.update(
            ids=batch_ids,
            embeddings=embeddings.tolist()
        )
        t_write += time.time() - t0

        n_processed += len(batch_ids)
        pbar.update(len(batch_ids))

    pbar.close()
    elapsed = time.time() - start

    print()
    print("EMBEDDING COMPLETE")
    print(f"  Chunks embedded: {n_processed:,}")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"  Throughput: {n_processed/elapsed:.0f} chunks/sec")
    print()
    
    print("STAGE BREAKDOWN")
    print(f"  Read  (collection.get):    {t_read:.1f}s  ({100*t_read/elapsed:.1f}%)")
    print(f"  Encode (model.encode):     {t_encode:.1f}s  ({100*t_encode/elapsed:.1f}%)")
    print(f"  Write (collection.update): {t_write:.1f}s  ({100*t_write/elapsed:.1f}%)")
    print(f"  Other (overhead/tqdm):     {elapsed - t_read - t_encode - t_write:.1f}s")
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Embedding batch size."
    )
    args = parser.parse_args()

    config = load_config(args.config)
    compute_embeddings(config, batch_size=args.batch_size)