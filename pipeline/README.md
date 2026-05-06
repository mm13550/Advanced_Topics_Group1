## Quickstart
 
### 1. Install dependencies
 
```bash
pip install -r requirements.txt
```
 
### 2. Download the database
 
```bash
python download_db.py
```
 
This will download the compressed archive (~2.33 GB) from HuggingFace, extract it to `data/chroma_db/`, and delete the compressed file automatically. No HuggingFace account is needed.
 
### 3. Verify the database
 
```bash
CUDA_VISIBLE_DEVICES="" python -m pipeline.smoke_test --week 2
```
 
All tests should pass. You will see output like:
 
```
PASS  Collection has documents  [677,786 chunks]
PASS  Year extraction failure < 20%
PASS  Query returns relevant results: ...
RESULT: ALL TESTS PASSED — Week 2 validation complete
```
 
---
 
## Notes
 
- **No GPU required** for downloading or running retrieval. Query embedding runs on CPU in milliseconds.
- **Do not re-run** `process_corpus.py` or `compute_embeddings.py` — these are only needed when rebuilding the database from scratch and take many hours.
- If `data/chroma_db/` already exists, `download_db.py` will skip the download. Delete that folder and rerun if you need a fresh copy.
- The `jurisdiction` metadata field is currently empty — this is a known limitation and is being worked on.