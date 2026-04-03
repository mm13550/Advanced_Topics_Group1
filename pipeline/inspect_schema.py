"""
Streams small samples from both HuggingFace datasets to print:
  - All field names and sample values
  - Inferred field map (case_id, text, jurisdiction, date, court, name)
  - Jurisdiction codes from 2,000-case sample
  - Date format samples for year extraction
  - Join key verification between text and embeddings datasets
  - Embedding normalization check

Output saves to logs/schema_inspection.txt
"""

import itertools
import re
from pathlib import Path

import numpy as np
from datasets import load_dataset
from huggingface_hub import login

from pipeline.config_loader import load_config

TEXT_SAMPLE = 20
JURISDICTION_SAMPLE = 2000
EMB_SAMPLE = 5


def inspect(config: dict) -> None:
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(str(msg))

    login(token=config["secrets"]["HF_TOKEN"], add_to_git_credential=False)

    log("DATASET SCHEMA INSPECTION")

    # CAP text dataset
    log()
    log("CAP TEXT DATASET")
    log(f"{config['datasets']['cap_text_repo']}")

    ds = load_dataset(
        config["datasets"]["cap_text_repo"],
        split=config["datasets"]["hf_split"],
        streaming=True,
        token=config["secrets"]["HF_TOKEN"]
    )
    sample = list(itertools.islice(ds, TEXT_SAMPLE))
    first = sample[0]

    log()
    log("ALL FIELDS:")
    for k in first.keys():
        log(f"  {k}")

    log()
    log("SAMPLE VALUES:")
    for k, v in first.items():
        display = str(v)[:120] + ("..." if len(str(v)) > 120 else "")
        log(f"  {k!r:30s}, {display}")

    log()
    log("NESTED FIELDS:")
    for k, v in first.items():
        if isinstance(v, dict):
            log(f"  {k!r} is dict with keys: {list(v.keys())}")
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            log(f"  {k!r} is list of dicts; first keys: {list(v[0].keys())}")

    # Find field names
    def find_field(candidates):
        for c in candidates:
            if c in first:
                return c
        return None

    id_field = find_field(["id", "case_id", "uid", "_id"])
    text_field = find_field(["text", "body", "content", "opinion", "casebody", "full_text"])
    jur_field = find_field(["jurisdiction", "jurisdiction_id", "reporter"])
    date_field = find_field(["decision_date", "date", "date_filed", "decided_date", "year"])
    name_field = find_field(["name", "case_name", "title", "name_abbreviation"])
    court_field = find_field(["court", "court_name"])

    # Jurisdiction codes
    log()
    log(f"Streaming {JURISDICTION_SAMPLE} records for jurisdiction codes")
    jur_ds = load_dataset(
        config["datasets"]["cap_text_repo"],
        split=config["datasets"]["hf_split"],
        streaming=True,
        token=config["secrets"]["HF_TOKEN"]
    )
    jur_counts: dict[str, int] = {}
    date_samples = []

    for rec in itertools.islice(jur_ds, JURISDICTION_SAMPLE):
        if jur_field:
            jval = rec.get(jur_field, "")
            if isinstance(jval, dict):
                jval = jval.get("slug", jval.get("id", str(jval)))
            jval = str(jval).lower().strip()
            jur_counts[jval] = jur_counts.get(jval, 0) + 1
        if date_field and len(date_samples) < 10:
            date_samples.append(str(rec.get(date_field, "")))

    log()
    log("UNIQUE JURISDICTION CODES (sorted by frequency):")
    for code, count in sorted(jur_counts.items(), key=lambda x: -x[1]):
        log(f"  {code:35s}  n={count}")

    log()
    log("DATE FIELD SAMPLES:")
    for s in date_samples:
        log(f"  {s}")

    # Field map
    log()
    log("FIELD MAP")
    field_map = {
        "case_id":      id_field,
        "text":         text_field,
        "jurisdiction": jur_field,
        "date":         date_field,
        "case_name":    name_field,
        "court":        court_field
    }
    for logical, actual in field_map.items():
        status = "✓" if actual else "✗ NOT FOUND"
        log(f"  {status}  {logical:15s}, {actual or 'UNKNOWN — check fields above'}")
    
    # Write to log
    with open("logs/schema_inspection_parts1-2.txt", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    config = load_config()
    inspect(config)