"""
RAG System for Legal Question Answering
Combines retrieval from case database with LLM generation using IRAC framework.
"""

import os
import re
import json
import sys
from datetime import datetime
from typing import List, Dict, Optional
from dotenv import load_dotenv

from pipeline.retrieval_chroma import ChromaDBRetrievalPipeline
from pipeline.query_decomposer import QueryDecomposer

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
        self.decomposer = QueryDecomposer()
        
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
        return f"""You are a legal research assistant. Answer ONLY from the retrieved case excerpts below.

    Retrieved cases:
    {context}

    Output format (must follow exactly):
    1. Start with `Final Answer:` on one line, followed by a direct 1-2 sentence answer.
    2. Then provide an IRAC analysis with these exact labels on separate lines:
       - `Issue:`
       - `Rule:`
       - `Application:`
       - `Conclusion:`

    Evidence and citation rules:
    1. Use only the retrieved cases; do not use outside knowledge.
    2. When citing support, use citations exactly as provided in context metadata.
    3. If quoting text from a case, keep quotes short and relevant.
    4. If the retrieved cases do not directly answer the question, explicitly say so in `Final Answer` and `Conclusion`.
    5. Never fabricate citations or case details.
    6. If the user's query contains exclusion instructions (e.g. "excluding race-based claims"),
    explicitly respect those constraints in your response. Do not discuss excluded topics
    even if retrieved cases mention them."""
    
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
        temperature: float = 0.3,
        chat_history: Optional[List[Dict]] = None
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
            # Decompose query into semantic query + metadata filters
            decomposed = self.decomposer.decompose(query)
            semantic_query = decomposed["semantic_query"]

            if decomposed["was_decomposed"]:
                print(f"Query decomposed: '{semantic_query}'")
                active_filters = {
                    k: v for k, v in decomposed.items()
                    if k in ("year_min", "year_max", "court_contains")
                    and v is not None
                }
                if active_filters:
                    print(f"Metadata filters: {active_filters}")
            
            retrieved_cases = self.retriever.retrieve(
                semantic_query,
                top_k=top_k,
                year_min=decomposed["year_min"],
                year_max=decomposed["year_max"],
                court_contains=decomposed["court_contains"]
            )
            
            if not retrieved_cases:
                print("Warning: No relevant cases found, falling back to IRAC-only mode")
                mode = "irac_only"
            else:
                context = self.retriever.format_context(retrieved_cases)
                system_prompt = self._build_irac_prompt(query, context)
            
        if mode == "irac_only":
            system_prompt = self._build_irac_only_prompt()
            
        else:
            system_prompt = self._build_baseline_prompt()
        
        messages = [{"role": "system", "content": system_prompt}]
        if chat_history:
            messages.extend(chat_history)
        messages.append({"role": "user", "content": query})

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )

            final_answer = completion.choices[0].message.content.strip()

            result = {
                "query": query,
                "response": final_answer,
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
