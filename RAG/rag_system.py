"""
RAG System for Legal Question Answering
Combines retrieval from case database with LLM generation using IRAC framework.
"""

import os
import json
import sys
from datetime import datetime
from typing import List, Dict, Optional
from dotenv import load_dotenv

from pipeline.retrieval_chroma import ChromaDBRetrievalPipeline

class RAGSystem:
    """
    Retrieval-Augmented Generation system for legal analysis.
    Retrieves relevant cases and generates IRAC-structured responses.
    """
    
    def __init__(
        self, 
        vector_db_path: str = "data/chroma_db",    # "vector_db.pkl",
        model_name: str = "llama-3.3-70b-versatile"
    ):
        """
        Initialize RAG system with retrieval and generation components.
        
        Args:
            vector_db_path: Path to vector database
            model_name: Groq model name
        """
        load_dotenv()
        
        #from retrieval import RetrievalPipeline
        from groq import Groq
        
        print("Initializing RAG system...")
        
        #self.retriever = RetrievalPipeline(vector_db_path)
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        self.retriever = ChromaDBRetrievalPipeline(vector_db_path)
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env file")
        
        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        self.conversation_history: List[Dict] = []
        
        print("RAG system ready")
    
    def _build_irac_prompt(self, query: str, context: str) -> str:
        """
        Build IRAC-structured system prompt with retrieved context.
        
        IRAC Framework:
        - Issue: Legal question or problem
        - Rule: Applicable legal principles
        - Application: Apply rules to facts
        - Conclusion: Final determination
        
        Args:
            query: User's legal question
            context: Retrieved case text
            
        Returns:
            Formatted system prompt
        """
        return f"""You are a legal research assistant. Answer the user's question using the IRAC framework (Issue, Rule, Application, Conclusion).

Use the following legal cases as supporting precedent:

{context}

Instructions:
1. Structure your response using IRAC:
   - Issue: State the legal question
   - Rule: Identify relevant legal principles from the provided cases
   - Application: Apply the rules to the question
   - Conclusion: Provide a clear answer

2. Cite specific cases when referencing legal principles
3. Be precise and professional
4. If the provided cases don't fully address the question, acknowledge this
5. Do not fabricate case citations beyond what is provided

Provide a thorough legal analysis based on the precedents given."""
    
    def _build_baseline_prompt(self) -> str:
        """Build system prompt for baseline (no IRAC, no retrieval)."""
        return (
            "You are a legal assistant. Answer the user's legal question "
            "clearly and accurately. Acknowledge uncertainty when appropriate."
        )
    
    def _build_irac_only_prompt(self) -> str:
        """Build IRAC prompt without retrieval context."""
        return """You are a legal research assistant. Answer the user's question using the IRAC framework (Issue, Rule, Application, Conclusion).

Structure your response as follows:
1. Issue: State the legal question
2. Rule: Identify relevant legal principles
3. Application: Apply the rules to the question
4. Conclusion: Provide a clear answer

Be precise, professional, and acknowledge if you're uncertain about specific precedents."""
    
    def generate_response(
        self,
        query: str,
        mode: str = "rag",
        top_k: int = 3,
        max_tokens: int = 800,
        temperature: float = 0.3
    ) -> Dict:
        """
        Generate response to legal query.
        
        Args:
            query: User's legal question
            mode: One of "rag" (full RAG), "irac_only" (IRAC without retrieval), 
                  or "baseline" (no IRAC, no retrieval)
            top_k: Number of cases to retrieve (only for RAG mode)
            max_tokens: Maximum response length
            temperature: Sampling temperature
            
        Returns:
            Dictionary with response, retrieved cases, and metadata
        """
        retrieved_cases = []
        
        if mode == "rag":
            retrieved_cases = self.retriever.retrieve(query, top_k=top_k)
            
            if not retrieved_cases:
                print("Warning: No relevant cases found, falling back to IRAC-only mode")
                mode = "irac_only"
        
        if mode == "rag":
            context = self.retriever.format_context(retrieved_cases)
            system_prompt = self._build_irac_prompt(query, context)
        elif mode == "irac_only":
            system_prompt = self._build_irac_only_prompt()
        else:
            system_prompt = self._build_baseline_prompt()
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            response = completion.choices[0].message.content
            
            result = {
                "query": query,
                "response": response,
                "mode": mode,
                "retrieved_cases": retrieved_cases,
                "num_cases_retrieved": len(retrieved_cases),
                "timestamp": datetime.now().isoformat(),
                "model": self.model_name
            }
            
            self.conversation_history.append(result)
            
            return result
            
        except Exception as e:
            return {
                "query": query,
                "response": f"Error: {str(e)}",
                "mode": mode,
                "retrieved_cases": [],
                "num_cases_retrieved": 0,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def save_history(self, filepath: str = "rag_history.json"):
        """Save conversation history to JSON."""
        with open(filepath, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)
    
    def load_history(self, filepath: str = "rag_history.json"):
        """Load conversation history from JSON."""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.conversation_history = json.load(f)


def compare_modes():
    """Compare responses across different modes."""
    test_query = "What are the elements required to prove negligence in a tort case?"
    
    print("Comparing RAG modes")
    print("=" * 60)
    print(f"Query: {test_query}\n")
    
    rag = RAGSystem()
    
    modes = ["baseline", "irac_only", "rag"]
    results = {}
    
    for mode in modes:
        print(f"\nMode: {mode.upper()}")
        print("-" * 60)
        
        result = rag.generate_response(test_query, mode=mode)
        results[mode] = result
        
        if mode == "rag" and result["retrieved_cases"]:
            print(f"Retrieved {result['num_cases_retrieved']} cases")
            print(f"Top similarity: {result['retrieved_cases'][0]['similarity']:.4f}\n")
        
        print(f"Response:\n{result['response']}\n")
        print("=" * 60)
    
    rag.save_history("mode_comparison.json")
    print("\nComparison saved to mode_comparison.json")


def run_test_queries():
    """Run test queries in RAG mode."""
    test_queries = [
        "What is probable cause for a search warrant?",
        "What are the requirements for a valid contract?",
    ]

# "When can a landlord evict a tenant without notice?"
    
    rag = RAGSystem()
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}/{len(test_queries)}")
        print(f"Query: {query}")
        print("-" * 60)
        
        result = rag.generate_response(query, mode="rag", top_k=3)
        
        print(f"\nRetrieved {result['num_cases_retrieved']} cases")
        
        if result['retrieved_cases']:
            print("Case similarities:", [
                f"{c['similarity']:.3f}" for c in result['retrieved_cases']
            ])
        
        print(f"\nResponse:\n{result['response']}\n")
        print("=" * 60)
    
    rag.save_history("test_rag_results.json")
    print("\nResults saved to test_rag_results.json")


def interactive_mode():
    """Interactive RAG interface."""
    print("RAG Legal Assistant")
    print("Type 'quit' to exit")
    print("Commands: 'mode:rag', 'mode:irac', 'mode:baseline'\n")
    
    rag = RAGSystem()
    current_mode = "rag"
    
    while True:
        try:
            user_input = input(f"[{current_mode}] Question: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                rag.save_history()
                break
            
            if user_input.lower().startswith("mode:"):
                new_mode = user_input.split(":")[1].strip()
                if new_mode in ["rag", "irac", "irac_only", "baseline"]:
                    current_mode = new_mode if new_mode != "irac" else "irac_only"
                    print(f"Mode changed to: {current_mode}")
                continue
            
            if not user_input:
                continue
            
            result = rag.generate_response(user_input, mode=current_mode)
            
            if current_mode == "rag" and result['retrieved_cases']:
                print(f"\n[Retrieved {result['num_cases_retrieved']} cases]")
            
            print(f"\n{result['response']}\n")
            
        except KeyboardInterrupt:
            rag.save_history()
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            run_test_queries()
        elif sys.argv[1] == "compare":
            compare_modes()
    else:
        interactive_mode()
