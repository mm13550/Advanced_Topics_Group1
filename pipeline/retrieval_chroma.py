"""
Retrieves from ChromaDB vector database

To switch databases in RAG/rag_system.py, change imports:
  ORIGINAL: from retrieval import RetrievalPipeline
  MODIFIED:from pipeline.retrieval_chroma import ChromaDBRetrievalPipeline as RetrievalPipeline

Uses a two-stage retrieval design:
1. Top n candidates retrieved using cosine similarity
2. Cross-encoder reranking of the top n candidates to produce final top k results
"""

import os
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional

from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from chromadb.config import Settings


class ChromaDBRetrievalPipeline:
    """
    ChromaDB-based retrieval pipeline with cross-encoder reranking

    Replaces RAG/retrieval.py's RetrievalPipeline
    """

    def __init__(
        self,
        vector_db_path: str = "data/chroma_db",
        collection_name: str = "legal_cases",
        use_reranker: bool = True,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """
        Initializes ChromaDB retrieval pipeline

        Args:
            vector_db_path:   Path to ChromaDB's directory
            collection_name:  ChromaDB collection name
            use_reranker:     Applies cross-encoder reranking
            reranker_model:   HuggingFace model ID for cross-encoder
        """
        print("Starting ChromaDB retrieval pipeline:")

        # Load embedding model
        from pipeline.config_loader import load_config
        config = load_config()
        self.embedding_model = SentenceTransformer(
            config["embeddings"]["model"],
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print("Loaded embedding model")


        # Connect to ChromaDB
        if not os.path.exists(vector_db_path):
            raise FileNotFoundError(
                f"ChromaDB directory not found at '{vector_db_path}'"
            )

        client = chromadb.PersistentClient(
            path=vector_db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection = client.get_collection(collection_name)
        n_chunks = self.collection.count()
        print(f"Connected to ChromaDB. {n_chunks:,} chunks in '{collection_name}'")


        # Cross-encoder reranker
        self.use_reranker = use_reranker
        self.reranker = None

        if use_reranker:
            try:
                self.reranker = CrossEncoder(reranker_model, device="cpu")
                print(f"Loaded reranker model: {reranker_model}")
            except Exception as e:
                print(f"  Warning: Could not load reranker ({e}). "
                      f"Falling back to cosine-similarity-only retrieval.")
                self.use_reranker = False

        print("ChromaDB retrieval pipeline ready.")


    # Public interface similar to RAG/retrieval.py
    def embed_query(self, query: str) -> np.ndarray:
        """
        Converts text query to 768-dimensional BGE embedding.

        Args:
            query: User's legal question.

        Returns:
            768-dim float32 numpy array, L2-normalized.
        """
        embedding = self.embedding_model.encode(
            "Represent this sentence for searching relevant passages: " + query,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embedding.astype(np.float32)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        min_similarity: float = 0.0,
        stage1_candidates: int = 20,
        year_min: int = None,
        year_max: int = None,
        court_contains: str = None
    ) -> List[Dict]:
        """
        Retrieves top k most relevant legal cases using two-stage retrieval.

        Steps:
        1. ChromaDB cosine similarity search
        2. Cross-encoder reranking

        Return format is identical to RetrievalPipeline.retrieve()

        Args:
            query: User's legal question
            top_k: Number of final results to return
            min_similarity: Minimum step 1 cosine similarity, defaults to 0.0/no filter.
                        EDA found ≥ 0.82 indicates genuine semantic relevance
            stage1_candidates: Number of candidates to retrieve in step 1
                        before reranking. Should be > top_k.
                        Ignored if use_reranker=False

        Returns:
            List of dicts, each with keys:
              text: full opinion text
              preview: first 500 chars for display
              similarity: cosine similarity score
              rerank_score: cross-encoder score, or None
              case_id: unique case identifier from CAP dataset
              case_name: extracted case name
              court: court name
              year: decision year
              below_floor: True if similarity < 0.82
        """
        # ChromaDB vector search
        query_embedding = self.embed_query(query)

        n_retrieve = stage1_candidates if self.use_reranker else top_k


        # Metadata filter
        where_clause = None
        conditions = []
        
        if year_min is not None:
            conditions.append({"year": {"$gte": year_min}})
        if year_max is not None:
            conditions.append({"year": {"$lte": year_max}})
        
        if len(conditions) == 1:
            where_clause = conditions[0]
        elif len(conditions) > 1:
            where_clause = {"$and": conditions}
        
        # ChromaDB query with optional filter
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(n_retrieve, self.collection.count()),
            include=["documents", "metadatas", "distances"],
            where=where_clause   # None means no filter
        )

        # ChromaDB returns cosine distance, so convert to similarity and apply min_similarity filter
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        stage1_results = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            similarity = float(1.0 - dist)
            if similarity < min_similarity:
                continue
            stage1_results.append({
                "text": doc,
                "preview": doc[:500],
                "similarity": similarity,
                "rerank_score": None,
                "case_id": meta.get("case_id", ""),
                "case_name": meta.get("case_name", "Unknown"),
                "court": meta.get("court", ""),
                "year": meta.get("year", 0),
                "below_floor": similarity < 0.82,
                "chunk_index": meta.get("chunk_index", 0)
            })
            
        # Post-filter by court name if specified
        if court_contains:
            stage1_results = [
                r for r in stage1_results
                if court_contains.lower() in r["court"].lower()
            ]

        if not stage1_results:
            return []

        # Cross-encoder reranking
        if self.use_reranker and self.reranker is not None and len(stage1_results) > 1:
            pairs = [(query, r["text"]) for r in stage1_results]
            rerank_scores = self.reranker.predict(pairs).tolist()

            for result, score in zip(stage1_results, rerank_scores):
                result["rerank_score"] = float(score)

            # Sort by rerank score descending, take top_k
            stage1_results.sort(key=lambda x: x["rerank_score"], reverse=True)

        return stage1_results[:top_k]

    def format_context(
        self,
        retrieved_cases: List[Dict],
        max_chars_per_case: int = 1000
    ) -> str:
        """
        Format retrieved cases into context string for LLM.

        Args:
            retrieved_cases: Output of retrieve()
            max_chars_per_case: Maximum characters of opinion text per case

        Returns:
            Formatted context string ready for LLM system prompt injection
        """
        if not retrieved_cases:
            return "No relevant cases found."

        context_parts = []
        for i, case in enumerate(retrieved_cases, 1):
            # Truncate opinion text to token budget
            case_text = case["text"][:max_chars_per_case]
            if len(case["text"]) > max_chars_per_case:
                case_text += "... [truncated]"

            # Build metadata header
            meta_parts = [f"Case {i} (similarity: {case['similarity']:.3f})"]
            if case.get("rerank_score") is not None:
                meta_parts.append(f"rerank: {case['rerank_score']:.3f}")
            if case.get("case_name") and case["case_name"] != "Unknown":
                meta_parts.append(f"name: {case['case_name']}")
            if case.get("court"):
                meta_parts.append(f"court: {case['court']}")
            if case.get("year"):
                meta_parts.append(f"year: {case['year']}")
            if case.get("below_floor"):
                meta_parts.append("⚠ below relevance threshold")

            header = " | ".join(meta_parts)
            context_parts.append(f"{header}:\n{case_text}\n")

        return "\n---\n".join(context_parts)

    def get_stats(self) -> Dict:
        """
        Return database statistics. Mirrors VectorDatabase.get_stats()
        so evaluation.py can call the same method on either backend.
        """
        n = self.collection.count()

        # Sample metadata for jurisdiction/year breakdown
        sample = self.collection.get(
            limit=min(500, n),
            include=["metadatas"],
        )
        metadatas = sample["metadatas"]

        year_counts: Dict[int, int] = {}
        court_counts: Dict[str, int] = {}
        below_floor_count = 0

        for meta in metadatas:
            y = meta.get("year", 0)
            if y:
                year_counts[y] = year_counts.get(y, 0) + 1
            c = meta.get("court", "")
            if c:
                court_counts[c] = court_counts.get(c, 0) + 1

        return {
            "num_chunks": n,
            "reranker_active": self.use_reranker,
            "year_range": (min(year_counts) if year_counts else None,
                        max(year_counts) if year_counts else None),
            "top_courts": sorted(court_counts.items(), key=lambda x: -x[1])[:5]
        }