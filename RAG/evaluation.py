"""
Evaluation Suite for RAG Legal Assistant
Compares baseline LLM vs RAG system across multiple metrics.
"""

import json
import re
from typing import List, Dict, Tuple
from datetime import datetime
import numpy as np


class RAGEvaluator:
    """
    Evaluates RAG system performance across multiple dimensions:
    - Citation accuracy
    - Response relevance
    - IRAC structure compliance
    - Baseline vs RAG comparison
    """
    
    def __init__(self, vector_db_path: str = "vector_db.pkl"):
        from rag_system import RAGSystem
        from baseline_llm import BaseLLM
        
        self.rag = RAGSystem(vector_db_path)
        self.baseline = BaseLLM()
        self.results = []
    
    def evaluate_citation_accuracy(self, response: str, retrieved_cases: List[Dict]) -> Dict:
        """
        Check if response cites real cases from retrieved set.
        
        Args:
            response: Generated response text
            retrieved_cases: List of retrieved case dictionaries
            
        Returns:
            Dictionary with citation metrics
        """
        case_references = re.findall(r'[Cc]ase\s+(\d+)', response)
        
        cited_cases = set(int(ref) for ref in case_references if int(ref) <= len(retrieved_cases))
        
        total_citations = len(case_references)
        valid_citations = len(cited_cases)
        
        if total_citations == 0:
            accuracy = 0.0
            hallucination_rate = 0.0
        else:
            accuracy = valid_citations / total_citations
            hallucination_rate = (total_citations - valid_citations) / total_citations
        
        return {
            "total_citations": total_citations,
            "valid_citations": valid_citations,
            "citation_accuracy": accuracy,
            "hallucination_rate": hallucination_rate,
            "cited_case_ids": list(cited_cases)
        }
    
    def evaluate_irac_structure(self, response: str) -> Dict:
        """
        Check if response follows IRAC framework.
        
        Args:
            response: Generated response text
            
        Returns:
            Dictionary with IRAC compliance metrics
        """
        response_lower = response.lower()
        
        irac_keywords = {
            "issue": ["issue", "question", "problem"],
            "rule": ["rule", "law", "principle", "statute", "precedent"],
            "application": ["application", "apply", "applies", "applying", "here"],
            "conclusion": ["conclusion", "therefore", "thus", "conclude"]
        }
        
        sections_found = {}
        for section, keywords in irac_keywords.items():
            sections_found[section] = any(kw in response_lower for kw in keywords)
        
        sections_present = sum(sections_found.values())
        compliance_score = sections_present / 4.0
        
        has_structure = all(sections_found.values())
        
        return {
            "sections_found": sections_found,
            "sections_present": sections_present,
            "compliance_score": compliance_score,
            "has_full_structure": has_structure
        }
    
    def evaluate_response_quality(self, response: str) -> Dict:
        """
        Assess basic response quality metrics.
        
        Args:
            response: Generated response text
            
        Returns:
            Dictionary with quality metrics
        """
        words = response.split()
        sentences = re.split(r'[.!?]+', response)
        
        return {
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "avg_sentence_length": len(words) / max(len(sentences), 1),
            "has_legal_terms": any(
                term in response.lower() 
                for term in ["court", "law", "statute", "precedent", "case"]
            )
        }
    
    def run_evaluation(
        self,
        test_queries: List[str],
        save_path: str = "evaluation_results.json"
    ) -> Dict:
        """
        Run comprehensive evaluation on test queries.
        
        Args:
            test_queries: List of legal questions to test
            save_path: Path to save results
            
        Returns:
            Dictionary with aggregated evaluation results
        """
        print(f"Running evaluation on {len(test_queries)} queries...")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\nEvaluating query {i}/{len(test_queries)}: {query[:50]}...")
            
            baseline_response = self.baseline.generate_response(query)
            
            rag_result = self.rag.generate_response(query, mode="rag", top_k=3)
            
            irac_result = self.rag.generate_response(query, mode="irac_only")
            
            baseline_eval = {
                "mode": "baseline",
                "query": query,
                "response": baseline_response,
                "quality": self.evaluate_response_quality(baseline_response),
                "irac_structure": self.evaluate_irac_structure(baseline_response),
                "timestamp": datetime.now().isoformat()
            }
            
            irac_eval = {
                "mode": "irac_only",
                "query": query,
                "response": irac_result["response"],
                "quality": self.evaluate_response_quality(irac_result["response"]),
                "irac_structure": self.evaluate_irac_structure(irac_result["response"]),
                "timestamp": datetime.now().isoformat()
            }
            
            rag_eval = {
                "mode": "rag",
                "query": query,
                "response": rag_result["response"],
                "num_cases_retrieved": rag_result["num_cases_retrieved"],
                "quality": self.evaluate_response_quality(rag_result["response"]),
                "irac_structure": self.evaluate_irac_structure(rag_result["response"]),
                "citations": self.evaluate_citation_accuracy(
                    rag_result["response"],
                    rag_result["retrieved_cases"]
                ),
                "timestamp": datetime.now().isoformat()
            }
            
            self.results.append({
                "query": query,
                "baseline": baseline_eval,
                "irac_only": irac_eval,
                "rag": rag_eval
            })
        
        aggregated = self._aggregate_results()
        
        full_results = {
            "summary": aggregated,
            "individual_results": self.results,
            "evaluation_date": datetime.now().isoformat(),
            "num_queries": len(test_queries)
        }
        
        with open(save_path, 'w') as f:
            json.dump(full_results, f, indent=2)
        
        print(f"\nEvaluation complete. Results saved to {save_path}")
        
        return full_results
    
    def _aggregate_results(self) -> Dict:
        """Aggregate results across all queries."""
        
        baseline_irac = [r["baseline"]["irac_structure"]["compliance_score"] for r in self.results]
        irac_only_irac = [r["irac_only"]["irac_structure"]["compliance_score"] for r in self.results]
        rag_irac = [r["rag"]["irac_structure"]["compliance_score"] for r in self.results]
        
        baseline_words = [r["baseline"]["quality"]["word_count"] for r in self.results]
        irac_only_words = [r["irac_only"]["quality"]["word_count"] for r in self.results]
        rag_words = [r["rag"]["quality"]["word_count"] for r in self.results]
        
        rag_citations = [r["rag"]["citations"]["total_citations"] for r in self.results]
        rag_citation_accuracy = [
            r["rag"]["citations"]["citation_accuracy"] 
            for r in self.results 
            if r["rag"]["citations"]["total_citations"] > 0
        ]
        
        return {
            "baseline": {
                "avg_irac_compliance": np.mean(baseline_irac),
                "avg_word_count": np.mean(baseline_words),
                "responses_with_legal_terms": sum(
                    r["baseline"]["quality"]["has_legal_terms"] for r in self.results
                ) / len(self.results)
            },
            "irac_only": {
                "avg_irac_compliance": np.mean(irac_only_irac),
                "avg_word_count": np.mean(irac_only_words),
                "responses_with_legal_terms": sum(
                    r["irac_only"]["quality"]["has_legal_terms"] for r in self.results
                ) / len(self.results)
            },
            "rag": {
                "avg_irac_compliance": np.mean(rag_irac),
                "avg_word_count": np.mean(rag_words),
                "avg_citations": np.mean(rag_citations),
                "avg_citation_accuracy": np.mean(rag_citation_accuracy) if rag_citation_accuracy else 0.0,
                "responses_with_legal_terms": sum(
                    r["rag"]["quality"]["has_legal_terms"] for r in self.results
                ) / len(self.results)
            }
        }
    
    def print_summary(self):
        """Print evaluation summary to console."""
        if not self.results:
            print("No results to summarize. Run evaluation first.")
            return
        
        summary = self._aggregate_results()
        
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        print("\nBASELINE (No IRAC, No Retrieval):")
        print(f"  IRAC Compliance: {summary['baseline']['avg_irac_compliance']:.2%}")
        print(f"  Avg Word Count: {summary['baseline']['avg_word_count']:.0f}")
        print(f"  Legal Terms: {summary['baseline']['responses_with_legal_terms']:.2%}")
        
        print("\nIRAC ONLY (IRAC, No Retrieval):")
        print(f"  IRAC Compliance: {summary['irac_only']['avg_irac_compliance']:.2%}")
        print(f"  Avg Word Count: {summary['irac_only']['avg_word_count']:.0f}")
        print(f"  Legal Terms: {summary['irac_only']['responses_with_legal_terms']:.2%}")
        
        print("\nRAG (IRAC + Retrieval):")
        print(f"  IRAC Compliance: {summary['rag']['avg_irac_compliance']:.2%}")
        print(f"  Avg Word Count: {summary['rag']['avg_word_count']:.0f}")
        print(f"  Avg Citations: {summary['rag']['avg_citations']:.1f}")
        print(f"  Citation Accuracy: {summary['rag']['avg_citation_accuracy']:.2%}")
        print(f"  Legal Terms: {summary['rag']['responses_with_legal_terms']:.2%}")
        
        print("\n" + "="*60)


def run_default_evaluation():
    """Run evaluation with default test queries."""
    
    test_queries = [
        "What is probable cause for a search warrant?",
        "What are the elements of negligence in tort law?",
        "When can a landlord evict a tenant without notice?",
        "What constitutes a valid contract?",
        "What is the difference between murder and manslaughter?",
        "What are Miranda rights?",
        "When does double jeopardy apply?",
        "What is the reasonable person standard?"
    ]
    
    evaluator = RAGEvaluator()
    results = evaluator.run_evaluation(test_queries)
    evaluator.print_summary()
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "custom":
        print("Enter test queries: ")
        queries = []
        while True:
            query = input("> ").strip()
            if not query:
                break
            queries.append(query)
        
        if queries:
            evaluator = RAGEvaluator()
            evaluator.run_evaluation(queries)
            evaluator.print_summary()
    else:
        run_default_evaluation()
