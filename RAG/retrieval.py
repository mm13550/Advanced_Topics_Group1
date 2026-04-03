"""
Retrieval Pipeline for Legal Case Search
Converts user queries to embeddings and retrieves relevant cases from vector database.
"""

import os
import numpy as np
from typing import List, Dict, Tuple
from dotenv import load_dotenv


class RetrievalPipeline:
    """
    Handles query embedding and vector similarity search.
    Bridges user questions to relevant legal cases.
    """
    
    def __init__(self, vector_db_path: str = "vector_db.pkl"):
        """
        Initialize retrieval pipeline with vector database.
        
        Args:
            vector_db_path: Path to saved vector database
        """
        from sentence_transformers import SentenceTransformer
        from vector_db_setup import VectorDatabase
        
        print("Loading retrieval pipeline...")
        
        self.embedding_model = SentenceTransformer('BAAI/bge-base-en-v1.5')
        print("Loaded embedding model: BAAI/bge-base-en-v1.5 (768-dim)")
        
        self.vector_db = VectorDatabase()
        self.vector_db.load(vector_db_path)
        print(f"Loaded vector database with {self.vector_db.index.ntotal:,} cases")
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Convert text query to 768-dimensional embedding.
        
        Args:
            query: User's legal question
            
        Returns:
            768-dimensional numpy array
        """
        embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embedding.astype(np.float32)
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 3,
        min_similarity: float = 0.0
    ) -> List[Dict]:
        """
        Retrieve most relevant legal cases for a query.
        
        Args:
            query: User's legal question
            top_k: Number of cases to retrieve
            min_similarity: Minimum cosine similarity threshold
            
        Returns:
            List of case dictionaries with text and metadata
        """
        query_embedding = self.embed_query(query)
        
        results = self.vector_db.search(query_embedding, top_k=top_k)
        
        filtered_results = [
            {
                "text": metadata["full_text"],
                "preview": metadata["text"],
                "similarity": score,
                "case_id": metadata["embedding_id"]
            }
            for score, metadata in results
            if score >= min_similarity
        ]
        
        return filtered_results
    
    def format_context(self, retrieved_cases: List[Dict], max_chars_per_case: int = 1000) -> str:
        """
        Format retrieved cases into context string for LLM.
        
        Args:
            retrieved_cases: List of retrieved case dictionaries
            max_chars_per_case: Maximum characters per case to avoid token limits
            
        Returns:
            Formatted context string
        """
        if not retrieved_cases:
            return "No relevant cases found."
        
        context_parts = []
        for i, case in enumerate(retrieved_cases, 1):
            case_text = case['text'][:max_chars_per_case]
            if len(case['text']) > max_chars_per_case:
                case_text += "... [truncated]"
            
            context_parts.append(
                f"Case {i} (similarity: {case['similarity']:.3f}):\n"
                f"{case_text}\n"
            )
        
        return "\n---\n".join(context_parts)


def test_retrieval():
    """Test retrieval pipeline with sample queries."""
    test_queries = [
        "What constitutes probable cause for a search?",
        "Requirements for a valid contract",
        "Difference between negligence and gross negligence"
    ]
    
    print("Initializing retrieval pipeline...")
    retriever = RetrievalPipeline()
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        
        results = retriever.retrieve(query, top_k=3)
        
        if not results:
            print("No results found")
            continue
        
        for i, case in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Similarity: {case['similarity']:.4f}")
            print(f"Preview: {case['preview'][:200]}...")
        
        print("\n" + "=" * 60)


def interactive_search():
    """Interactive search interface."""
    print("Legal Case Search")
    print("Type 'quit' to exit\n")
    
    retriever = RetrievalPipeline()
    
    while True:
        try:
            query = input("Search query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            results = retriever.retrieve(query, top_k=3)
            
            print(f"\nFound {len(results)} relevant cases:\n")
            
            for i, case in enumerate(results, 1):
                print(f"{i}. Similarity: {case['similarity']:.4f}")
                print(f"   {case['preview'][:150]}...\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_retrieval()
    else:
        interactive_search()