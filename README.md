# RAG Legal Assistant

A conversational legal research tool that answers U.S. legal questions strictly from real court opinions — no hallucinated citations, no general knowledge fill-in. Built for Advanced Topics in Data Science (Group 1, NYUAD).

---

## How It Works (End to End)

```
Your question
    ↓
[1] BGE Embedding  (BAAI/bge-base-en-v1.5, 768-dim)
    ↓
[2] ChromaDB Vector Search  (cosine similarity, top-20 candidates)
    ↓
[3] Cross-Encoder Reranking  (ms-marco-MiniLM-L-6-v2, top-k results)
    ↓
[4] IRAC Prompt + Llama 3.3 70B via Groq
    (must quote verbatim from retrieved text, no speculation)
    ↓
[5] Conversational response  (full chat history passed each turn)
```

**IRAC Framework:** The LLM is forced to structure every answer as:
- **Issue** — identify the legal question
- **Rule** — state the legal principle drawn from retrieved cases
- **Application** — apply it to the question using verbatim quotes
- **Conclusion** — one direct answer based only on what the cases say

If the retrieved documents don't contain a clear answer, the model is required to say so rather than speculate.

---

## Features

| Feature | Detail |
|---|---|
| Conversational memory | Full chat history sent to the LLM each turn — follow-up questions work naturally |
| Speech-to-text | Click the mic, speak, stop — Google Speech Recognition fills the text box |
| Text-to-speech | Click 🔊 Hear response — generates and plays the answer as audio (Google TTS) |
| Retrieved Cases panel | Shows which court opinion chunks were used, similarity scores, and the text excerpt the model read |
| IRAC enforcement | System prompt requires structured legal reasoning with verbatim citations |
| Relevance threshold | Cases with similarity < 0.60 are flagged with a warning |

---

## Project Structure

```
.
├── RAG/
│   ├── rag_system.py          # Core RAG engine (retrieval + IRAC generation)
│   ├── baseline_llm.py        # Baseline LLM without retrieval (for comparison)
│   ├── web_interface.py       # Gradio chat interface
│   ├── evaluation.py          # Evaluation metrics & benchmarking
│   └── retrieval.py           # Legacy FAISS retrieval (superseded by ChromaDB)
├── pipeline/
│   ├── retrieval_chroma.py    # ChromaDB retrieval + cross-encoder reranking
│   ├── build_index.py         # Build ChromaDB index from corpus
│   ├── compute_embeddings.py  # Generate BGE embeddings
│   ├── process_corpus.py      # Preprocess raw case data
│   └── smoke_test.py          # Validate retrieval pipeline
├── configs/
│   └── config.yaml            # Embedding model + pipeline settings
├── data/                      # ChromaDB index lives here (not in git — download separately)
├── download_db.py             # Downloads ChromaDB from HuggingFace (~1.6 GB)
├── requirements.txt
└── .env                       # Your API keys (never commit this)
```

---

## Running Locally

### 1. Clone the repo

```bash
git clone https://github.com/mm13550/Advanced_Topics_Group1.git
cd Advanced_Topics_Group1
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install SpeechRecognition gTTS
```

> If you get a `click` version conflict after installing gTTS, run:
> `pip install "click>=8.2.1"`

### 4. Set up your API keys

Create a file called `.env` in the project root:

```
GROQ_API_KEY=your_groq_key_here
```

- **Groq API key** (free): [console.groq.com](https://console.groq.com) → API Keys → Create
- Each team member needs their own key — the free tier is sufficient
- No HuggingFace account or token needed — the ChromaDB dataset is public
- No LangSmith key needed — it is not used by this project

### 5. Download the ChromaDB index

```bash
python download_db.py
```

This downloads ~1.6 GB from HuggingFace and extracts it to `data/chroma_db/`. Takes a few minutes depending on your connection. Only needs to be done once.

### 6. Launch the app

```bash
python RAG/web_interface.py
```

Open **http://127.0.0.1:7860** in your browser.

---

## Using the Interface

- **Ask a question** — type in the box and press Enter or click the orange ↑ button
- **Voice input** — click the microphone, speak your question, stop recording. The transcript appears in the text box for review before you send.
- **Hear the response** — click **🔊 Hear response** after any answer to hear it read aloud
- **Retrieved Cases** — expand the accordion at the bottom to see which court opinions were used, their similarity scores (sorted highest → lowest), and the text chunk the model read
- **Settings** (collapsed by default):
  - *Cases to Retrieve* (1–5): how many reranked opinion chunks are fed to the LLM
  - *Creativity (Temperature)*: keep low (0.2–0.4) for legal research; higher = more varied phrasing
- **Clear conversation** — resets the full chat history and audio player

---

## Tech Stack

| Component | Tool |
|---|---|
| LLM | Llama 3.3 70B Versatile (Groq API) |
| Embeddings | BAAI/bge-base-en-v1.5 (768-dim) |
| Vector store | ChromaDB (persistent, cosine distance) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Web interface | Gradio 6 |
| Speech-to-text | Google Speech Recognition (via SpeechRecognition) |
| Text-to-speech | Google TTS (gTTS) |
| Dataset | Caselaw Access Project — 2M+ U.S. court opinions; 1,349 chunks indexed |

---

## Evaluation

```bash
python RAG/evaluation.py
```

Metrics tracked:
- **IRAC Compliance** — presence of Issue / Rule / Application / Conclusion sections
- **Citation Accuracy** — valid case references drawn from the retrieved set
- **Hallucination Rate** — citations that don't appear in retrieved documents
- **Response Quality** — word count, legal terminology density

Results saved to `RAG/evaluation_results.json`.

---

## Team

Advanced Topics in Data Science — Group 1, NYUAD Spring 2025

- Merry Ma — mm13550@nyu.edu
- Salma Mansour — sim8654@nyu.edu
- Hanwen Zhang — hz3177@nyu.edu
