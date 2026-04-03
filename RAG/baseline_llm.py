"""
Baseline LLM for Legal Question Answering (No Retrieval)
Uses Groq API for fast inference with Llama models.
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Optional
from dotenv import load_dotenv


class BaseLLM:
    """
    Baseline language model without retrieval augmentation.
    Serves as comparison baseline for RAG evaluation.
    """
    
    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        load_dotenv()
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env file")
        
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("Install groq: pip install groq")
        
        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        self.conversation_history: List[Dict] = []
    
    def generate_response(
        self, 
        query: str, 
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> str:
        """
        Generate response to legal query without retrieval.
        
        Args:
            query: User's legal question
            system_prompt: Optional system instructions
            max_tokens: Maximum response length
            temperature: Sampling temperature (0.0-1.0)
            
        Returns:
            Generated response text
        """
        if system_prompt is None:
            system_prompt = (
                "You are a legal assistant. Answer the user's legal question "
                "clearly and accurately. Acknowledge uncertainty when appropriate."
            )
        
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
            
            self.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "response": response,
                "model": self.model_name
            })
            
            return response
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def save_history(self, filepath: str = "baseline_history.json"):
        """Save conversation history to JSON."""
        with open(filepath, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)
    
    def load_history(self, filepath: str = "baseline_history.json"):
        """Load conversation history from JSON."""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.conversation_history = json.load(f)


def run_test_queries():
    """Run test queries to verify baseline LLM functionality."""
    test_queries = [
        "What is the difference between murder and manslaughter?",
        "Can a landlord evict a tenant without notice in California?",
        "What are the elements of a valid contract?"
    ]
    
    llm = BaseLLM()
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}/{len(test_queries)}")
        print(f"Query: {query}\n")
        
        response = llm.generate_response(query, max_tokens=300)
        print(f"Response:\n{response}\n")
        print("-" * 60)
    
    llm.save_history("test_baseline_results.json")
    print("\nResults saved to test_baseline_results.json")


def interactive_mode():
    """Interactive command-line interface."""
    print("Baseline LLM - Legal Question Answering")
    print("Type 'quit' to exit\n")
    
    llm = BaseLLM()
    
    while True:
        try:
            query = input("Question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                llm.save_history()
                break
            
            if not query:
                continue
            
            response = llm.generate_response(query)
            print(f"\n{response}\n")
            
        except KeyboardInterrupt:
            llm.save_history()
            break


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_test_queries()
    else:
        interactive_mode()
