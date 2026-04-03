"""
Validates the ChromaDB collection at any stage of the pipeline.

To run:
  python pipeline/smoke_test.py     # full test
  python pipeline/smoke_test.py --week 1     # only text/metadata checks
  python pipeline/smoke_test.py --week 2     # includes embedding checks
"""

import argparse
import sys
from pathlib import Path

from pipeline.config_loader import load_config
from pipeline.process_corpus import get_chroma_collection

from langchain_community.embeddings import HuggingFaceEmbeddings

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"

def fetch_all_metadatas(collection, batch_size=5000):
    all_ids = collection.get(include=[])["ids"]
    all_meta = []
    for i in range(0, len(all_ids), batch_size):
        batch_ids = all_ids[i:i + batch_size]
        batch = collection.get(ids=batch_ids, include=["metadatas"])
        all_meta.extend(batch["metadatas"])
    return all_meta

def run(config: dict, week: int) -> bool:
    lines = []
    all_passed = True

    def log(msg=""):
        print(msg)
        lines.append(str(msg))

    def check(cond: bool, label: str, detail: str = "") -> bool:
        nonlocal all_passed
        tag = PASS if cond else FAIL
        log(f"{tag}  {label}" + (f"  [{detail}]" if detail else ""))
        if not cond:
            all_passed = False
        return cond

    log(f"SMOKE TEST — Week {week} validation")

    # Collection health
    log()
    log("Collection health")

    try:
        collection = get_chroma_collection(config)
    except Exception as e:
        check(False, "ChromaDB collection accessible", str(e))
        log("FATAL: Cannot open collection.")
        return False

    total = collection.count()
    check(total > 0, "Collection has documents", f"{total:,} chunks")

    # Spot check metadata on 5 random chunks
    sample = collection.get(limit=5, include=["documents", "metadatas"])
    for meta, doc in zip(sample["metadatas"], sample["documents"]):
        has_required = all(
            k in meta for k in ["case_id", "case_name", "jurisdiction", "year", "court", "chunk_index"]
        )
        check(has_required, f"Metadata complete for chunk {meta.get('chunk_id','?')}",
              str({k: meta.get(k) for k in ["jurisdiction", "year", "chunk_index"]}))
        check(len(doc) > 50, "Chunk text is non-trivial", f"{len(doc)} chars")

    all_meta = fetch_all_metadatas(collection)

    # Year extraction
    log()
    log("Year extraction")
    zero_year = sum(1 for m in all_meta if m.get("year", 0) == 0)
    pct_zero = 100 * zero_year / max(len(all_meta), 1)
    check(pct_zero < 20, "Year extraction failure < 20%", f"{pct_zero:.1f}% failed")

    years = [m["year"] for m in all_meta if m.get("year", 0) > 0]
    if years:
        log(f"  Year range: {min(years)} – {max(years)}")

    # Embeddings
    if week >= 2:
        log()
        log("Embedding check")

        import numpy as np

        # Small test queries
        test_queries = [
            "negligence duty of care landlord tenant",
            "First Amendment free speech government restriction",
            "contract breach damages expectation interest"
        ]

        embeddings = HuggingFaceEmbeddings(
            model_name=config["embeddings"]["model"]
        )

        for query in test_queries:
            results = collection.query(
                query_embeddings=[embeddings.embed_query(query)],
                n_results=3,
                include=["metadatas", "distances"]
            )
            top_score = 1 - results["distances"][0][0]
            top_meta = results["metadatas"][0][0]
            check(
                top_score >= 0.70,
                f"Query returns relevant results: '{query[:40]}...'",
                f"top score={top_score:.3f}, case={top_meta.get('case_name','?')[:40]}"
            )
            if top_score >= 0.82:
                log(f"Above relevance floor (0.82): {top_score:.3f}")
            else:
                log(f"Below relevance floor (0.82): {top_score:.3f}")

    log()
    if all_passed:
        log(f"RESULT: ALL TESTS PASSED — Week {week} validation complete")
    else:
        log("RESULT: SOME TESTS FAILED — review output above")
    log()

    # Save log
    Path("logs").mkdir(exist_ok=True)
    with open(f"logs/smoke_test_w{week}.txt", "w") as f:
        f.write("\n".join(lines))
    log(f"\nSaved to: logs/smoke_test_w{week}.txt")
    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--week", type=int, default=1, choices=[1, 2])
    args = parser.parse_args()
    cfg = load_config(args.config)
    passed = run(cfg, args.week)
    sys.exit(0 if passed else 1)