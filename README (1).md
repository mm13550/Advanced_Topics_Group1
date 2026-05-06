# RAG Legal Assistant

Retrieval-Augmented Generation system for legal case analysis using IRAC framework.

## System Overview

```
User Question
    ↓
[1] Baseline LLM (no retrieval)
    ↓
[2] Chroma Retrieval Pipeline (search case database)
    ↓
[3] RAG System (retrieval + IRAC generation)
    ↓
[4] Gradio Web Interface / Evaluation
```

## Files

| File | Purpose | Usage |
|------|---------|-------|
| `baseline_llm.py` | Simple LLM without retrieval | `python baseline_llm.py test` |
| `pipeline/retrieval_chroma.py` | ChromaDB retrieval + reranking | used by `RAG/rag_system.py` |
| `pipeline/smoke_test.py` | Validate the Chroma database and semantic retrieval | `python -m pipeline.smoke_test --week 2` |
| `RAG/rag_system.py` | Full RAG with IRAC prompting | `python -m RAG.rag_system compare` |
| `RAG/web_interface.py` | Gradio web interface | `python -m RAG.web_interface` |
| `evaluation.py` | Metrics & comparison | `python evaluation.py` |

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup Environment
Create `.env` file:
```
GROQ_API_KEY=your_api_key_here
```

### 3. Build Vector Database
```bash
python download_db.py
```
Downloads and extracts the Chroma database into `data/chroma_db/`.

### 4. Test Components

**Baseline LLM:**
```bash
python baseline_llm.py test
```

**Retrieval smoke test:**
```bash
python -m pipeline.smoke_test --week 2
```

**RAG System:**
```bash
python -m RAG.rag_system compare
```

### 5. Launch Web Interface
```bash
python -m RAG.web_interface
```
Opens at http://127.0.0.1:7860

### 6. Run Evaluation
```bash
python evaluation.py
```
Generates `evaluation_results.json`


## Architecture

**Vector Database:**
- 768-dimensional BGE embeddings
- ChromaDB cosine similarity search with cross-encoder reranking
- Caselaw Access Project dataset

**Retrieval:**
- Query → embedding (BGE-base-en-v1.5)
- Search top-k similar cases
- Truncate to 1000 chars/case

**Generation:**
- Model: Llama 3.3 70B (Groq API)
- IRAC framework prompting
- Temperature: 0.3 (default)
- Max tokens: 800

## Evaluation Metrics

- **IRAC Compliance**: Presence of Issue/Rule/Application/Conclusion
- **Citation Accuracy**: Valid case references from retrieved set
- **Response Quality**: Word count, legal terminology usage
- **Hallucination Rate**: Invalid citations

## Example Workflow

```python
from rag_system import RAGSystem

rag = RAGSystem()

result = rag.generate_response(
    "What constitutes probable cause?",
    mode="rag",
    top_k=5
)

print(result["response"])
print(f"Retrieved {result['num_cases_retrieved']} cases")
```

## Dataset

**Source:** [Caselaw Access Project](https://huggingface.co/datasets/free-law/Caselaw_Access_Project_embeddings)
- 2M+ U.S. court opinions
- Federal and state jurisdictions
- 768-D BGE embeddings (first 512 tokens)

## Team

Advanced Topics in Data Science - Group 1
- Merry Ma (mm13550@nyu.edu)
- Salma Mansour (sim8654@nyu.edu)
- Hanwen Zhang (hz3177@nyu.edu)
