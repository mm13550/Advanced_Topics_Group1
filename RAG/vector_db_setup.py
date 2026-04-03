"""
Vector Database Setup for Caselaw Access Project Embeddings
Loads pre-computed 768-D embeddings and builds FAISS index.
"""

import os
import numpy as np
import pickle
import requests
import time
from typing import List, Dict, Tuple


class VectorDatabase:
    """FAISS-based vector database for legal case embeddings."""
    
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.index = None
        self.metadata = []
        self.embeddings = None
        
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            raise ImportError("Install faiss: pip install faiss-cpu")
    
    def load_from_huggingface(
        self, 
        sample_size: int = 10000, 
        random_seed: int = 42
    ):
        """
        Load embeddings from HuggingFace dataset via Rows API.
        
        Args:
            sample_size: Number of cases to load
            random_seed: Random seed for reproducibility
        """
        print(f"Loading {sample_size:,} cases from HuggingFace dataset...")
        
        DATASET = "free-law/Caselaw_Access_Project_embeddings"
        SPLIT = "train"
        TOTAL_ROWS = 2_000_000
        BATCH_SIZE = 100
        API_URL = f"https://datasets-server.huggingface.co/rows?dataset={DATASET}&config=default&split={SPLIT}"
        
        np.random.seed(random_seed)
        n_batches = (sample_size // BATCH_SIZE) + 1
        random_offsets = sorted(
            np.random.randint(0, TOTAL_ROWS - BATCH_SIZE, size=n_batches)
        )
        
        print(f"Sampling {n_batches} random offsets across {TOTAL_ROWS:,} total rows")
        
        records = []
        
        for offset in random_offsets:
            if len(records) >= sample_size:
                break
            
            url = f"{API_URL}&offset={int(offset)}&length={BATCH_SIZE}"
            response = requests.get(url, timeout=60)
            
            if response.status_code == 429:
                if records:
                    print(f"\nRate limited after {len(records):,} rows - using what we have.")
                    break
                print(f"\nRate limited - waiting 30 seconds...")
                time.sleep(30)
                continue
            
            if response.status_code != 200:
                print(f"\nAPI error {response.status_code} at offset {offset}, skipping.")
                continue
            
            data = response.json()
            rows = data.get("rows", [])
            for r in rows:
                row = r["row"]
                records.append({"text": row["text"], "embeddings": row["embeddings"]})
            
            print(f"Fetched {len(records):,} / {sample_size:,} rows (offset {offset:,})", end="\r")
            time.sleep(0.5)
        
        if len(records) == 0:
            raise ValueError(
                "Failed to load any embeddings. Possible causes:\n"
                "1. HuggingFace API rate limit\n"
                "2. Network connectivity issue\n"
                "3. Dataset temporarily unavailable\n"
                "Try again in a few minutes or use smaller sample_size"
            )
        
        embeddings_list = [r["embeddings"] for r in records]
        self.embeddings = np.array(embeddings_list, dtype=np.float32)
        
        self.metadata = [
            {
                "text": r["text"][:500],
                "full_text": r["text"],
                "embedding_id": i
            }
            for i, r in enumerate(records)
        ]
        
        print(f"\nLoaded {len(self.embeddings):,} embeddings")
        print(f"Shape: {self.embeddings.shape}")
        print(f"Memory: {self.embeddings.nbytes / 1024 / 1024:.2f} MB")
        
        self._build_index()
    
    def _build_index(self):
        """Build FAISS index from loaded embeddings."""
        print("Building FAISS index...")
        
        self.faiss.normalize_L2(self.embeddings)
        self.index = self.faiss.IndexFlatIP(self.dimension)
        self.index.add(self.embeddings)
        
        print(f"Index built with {self.index.ntotal:,} vectors")
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 5
    ) -> List[Tuple[float, Dict]]:
        """
        Search for most similar cases using cosine similarity.
        
        Args:
            query_embedding: Query vector (768-dim)
            top_k: Number of results to return
            
        Returns:
            List of (similarity_score, metadata) tuples
        """
        if self.index is None:
            raise ValueError("Index not built. Load data first.")
        
        query = query_embedding.reshape(1, -1).astype(np.float32)
        self.faiss.normalize_L2(query)
        
        scores, indices = self.index.search(query, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append((float(score), self.metadata[idx]))
        
        return results
    
    def save(self, filepath: str = "vector_db.pkl"):
        """Save vector database to disk."""
        data = {
            "embeddings": self.embeddings,
            "metadata": self.metadata,
            "dimension": self.dimension
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        index_path = filepath.replace(".pkl", "_index.faiss")
        self.faiss.write_index(self.index, index_path)
        
        print(f"\nSaved to {filepath} and {index_path}")
    
    def load(self, filepath: str = "vector_db.pkl"):
        """Load vector database from disk."""
        print(f"Loading from {filepath}...")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.embeddings = data["embeddings"]
        self.metadata = data["metadata"]
        self.dimension = data["dimension"]
        
        index_path = filepath.replace(".pkl", "_index.faiss")
        self.index = self.faiss.read_index(index_path)
        
        print(f"Loaded {len(self.embeddings):,} embeddings")
    
    def get_stats(self) -> Dict:
        """Compute database statistics."""
        if self.embeddings is None:
            return {"status": "empty"}
        
        norms = np.linalg.norm(self.embeddings, axis=1)
        
        sample_size = min(100, len(self.embeddings))
        sample_indices = np.random.choice(
            len(self.embeddings), sample_size, replace=False
        )
        sample_embeddings = self.embeddings[sample_indices].copy()
        
        self.faiss.normalize_L2(sample_embeddings)
        similarities = np.dot(sample_embeddings, sample_embeddings.T)
        triu_indices = np.triu_indices(sample_size, k=1)
        pairwise_sims = similarities[triu_indices]
        
        return {
            "num_vectors": len(self.embeddings),
            "dimension": self.dimension,
            "mean_l2_norm": float(norms.mean()),
            "std_l2_norm": float(norms.std()),
            "mean_pairwise_similarity": float(pairwise_sims.mean()),
            "std_pairwise_similarity": float(pairwise_sims.std()),
            "memory_mb": self.embeddings.nbytes / 1024 / 1024
        }


def test_search():
    """Test vector database search functionality."""
    print("Testing vector database search\n")
    
    db = VectorDatabase()
    
    if not os.path.exists("vector_db.pkl"):
        print("No database found. Run setup first:")
        print("python vector_db_setup.py --sample_size 1000")
        return
    
    db.load("vector_db.pkl")
    
    test_query = np.random.randn(768).astype(np.float32)
    results = db.search(test_query, top_k=5)
    
    print("Top 5 Results:\n")
    for i, (score, metadata) in enumerate(results, 1):
        print(f"Rank {i} | Similarity: {score:.4f}")
        print(f"Text: {metadata['text'][:150]}...\n")
    
    stats = db.get_stats()
    print("\nDatabase Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


def main():
    """Setup vector database from command line."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample_size", 
        type=int, 
        default=5000,
        help="Number of cases to load"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="vector_db.pkl",
        help="Output filepath"
    )
    parser.add_argument(
        "--test", 
        action="store_true",
        help="Test after loading"
    )
    
    args = parser.parse_args()
    
    db = VectorDatabase()
    db.load_from_huggingface(sample_size=args.sample_size)
    
    stats = db.get_stats()
    print("\nStatistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    db.save(args.output)
    
    if args.test:
        test_search()


if __name__ == "__main__":
    import sys
    
    if "--test-only" in sys.argv:
        test_search()
    else:
        main()