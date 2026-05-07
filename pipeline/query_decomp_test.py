"""
End-to-end retrieval quality test for ChromaDB pipeline with query decomposition.

Tests the full flow: query decomposition -> retrieval -> context formatting
Saves results to logs/retrieval_test.json for analysis.

To run:
    python3 -m pipeline.retrieval_test
    python3 -m pipeline.retrieval_test --top-k 5
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from pipeline.config_loader import load_config
from pipeline.retrieval_chroma import ChromaDBRetrievalPipeline
from pipeline.query_decomposer import QueryDecomposer

# Test queries covering different decomposition scenarios
TEST_QUERIES = [
    {
        "id": 1,
        "description": "Purely semantic, no filters",
        "query": "Can police search my house without a warrant?"
    },
    {
        "id": 2,
        "description": "Semantic with year filter",
        "query": "Find cases involving freedom of speech decided after 1960"
    },
    {
        "id": 3,
        "description": "Semantic with court and year filters",
        "query": "Find Supreme Court contract cases between 1950 and 1970"
    },
    {
        "id": 4,
        "description": "Purely semantic, complex legal concept",
        "query": "What are the elements required to prove negligence?"
    },
    {
        "id": 5,
        "description": "Semantic with exclusion instruction (no metadata equivalent)",
        "query": "Find discrimination cases excluding race-based claims"
    }
]


def run_retrieval_test(top_k: int = 3) -> bool:
    """
    Runs retrieval test across all test queries and saves results.

    Args:
        top_k: Number of cases to retrieve per query

    Returns:
        True if all queries retrieved at least one result
    """
    Path("logs").mkdir(exist_ok=True)

    print("RETRIEVAL TEST")
    print(f"  Top-k: {top_k}")
    print(f"  Queries: {len(TEST_QUERIES)}")
    print()

    # Initialize components
    config = load_config()
    retriever = ChromaDBRetrievalPipeline(
        vector_db_path=config["chroma"]["persist_directory"],
        collection_name=config["chroma"]["collection_name"]
    )
    decomposer = QueryDecomposer()

    all_results = []
    all_passed = True

    for test in TEST_QUERIES:
        print(f"Query {test['id']}: {test['description']}")
        print(f"  Input:  {test['query']}")

        # Decompose query
        decomposed = decomposer.decompose(test["query"])
        semantic_query = decomposed["semantic_query"]
        active_filters = {
            k: v for k, v in decomposed.items()
            if k in ("year_min", "year_max", "court_contains")
            and v is not None
        }

        print(f"  Semantic query: '{semantic_query}'")
        if active_filters:
            print(f"  Filters: {active_filters}")
        else:
            print(f"  Filters: none")

        # Retrieve
        retrieved = retriever.retrieve(
            semantic_query,
            top_k=top_k,
            year_min=decomposed["year_min"],
            year_max=decomposed["year_max"],
            court_contains=decomposed["court_contains"]
        )

        # Evaluate
        passed = len(retrieved) > 0
        if not passed:
            all_passed = False

        status = "PASS" if passed else "FAIL"
        print(f"  {status}  Retrieved {len(retrieved)} cases")

        if retrieved:
            top = retrieved[0]
            print(f"  Top case:    {top['case_name']}")
            print(f"  Citation:    {top['citation']}")
            print(f"  Similarity:  {top['similarity']:.3f}")
            if top.get("rerank_score") is not None:
                print(f"  Rerank:      {top['rerank_score']:.3f}")
            print(f"  Year:        {top['year']}")
            print(f"  Quote:       {top['relevance_quote'][:120]}...")
            print(f"  Below floor: {top['below_floor']}")

        print()

        # Store full result for JSON output
        all_results.append({
            "id": test["id"],
            "description": test["description"],
            "raw_query": test["query"],
            "decomposed": decomposed,
            "active_filters": active_filters,
            "passed": passed,
            "num_retrieved": len(retrieved),
            "retrieved_cases": [
                {
                    "case_name": r["case_name"],
                    "citation": r["citation"],
                    "similarity": round(r["similarity"], 4),
                    "rerank_score": round(r["rerank_score"], 4) if r.get("rerank_score") is not None else None,
                    "year": r["year"],
                    "court": r["court"],
                    "below_floor": r["below_floor"],
                    "relevance_quote": r["relevance_quote"]
                }
                for r in retrieved
            ]
        })

    # Summary
    n_passed = sum(1 for r in all_results if r["passed"])
    print("=" * 60)
    print(f"RESULT: {n_passed}/{len(TEST_QUERIES)} queries retrieved results")
    if all_passed:
        print("ALL QUERIES RETURNED RESULTS")
    else:
        print("SOME QUERIES RETURNED NO RESULTS — review filters")
    print()

    # Save to JSON
    output = {
        "timestamp": datetime.now().isoformat(),
        "top_k": top_k,
        "num_queries": len(TEST_QUERIES),
        "num_passed": n_passed,
        "results": all_results
    }

    log_path = "logs/retrieval_test.json"
    with open(log_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to: {log_path}")

    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieval quality test")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    passed = run_retrieval_test(top_k=args.top_k)
    sys.exit(0 if passed else 1)