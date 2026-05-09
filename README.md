# Financial RAG Chatbot

A **Retrieval-Augmented Generation (RAG)** application that answers questions about a company's financial statements (last two years), built with open-source models only.

Live components:
- Sentence-aware chunking with abbreviation/decimal protection (handles `Alphabet Inc.`, `$309.1 billion` etc.)
- Hybrid retrieval: FAISS dense (`all-MiniLM-L6-v2`) + BM25 sparse, weighted by query-adaptive alpha
- Cross-encoder reranker (`BAAI/bge-reranker-base`) for confidence calibration
- Answer synthesis with `google/flan-t5-large` (770M params)
- Input guardrail (financial-relevance check) + three-band confidence labels (High / Medium / Low)
- Streamlit UI with file upload, chat history, source inspection

## Assignment scope

| Component | Implementation |
|---|---|
| Data Collection & Preprocessing | Last two years of Google's 10-K (`fin_stmts/`); pdfplumber extraction; sentence-aware chunking |
| Basic RAG | SentenceTransformer embeddings + FAISS `IndexFlatIP` (cosine via inner product) |
| UI (Streamlit) | Chat interface, file upload, confidence banner, expandable source view, retrieval-parameters panel |
| Advanced RAG | BM25 hybrid retrieval, chunk merging, adaptive k/alpha selection, BGE cross-encoder reranking |
| Guardrail | Input-side: `validate_query` rejects empty / too-short / non-financial queries |
| Testing | Three-category battery (high-conf, low-conf, irrelevant) in `tests/test_e2e.py` |

Advanced RAG technique: **Chunk merging + adaptive retrieval**, plus cross-encoder reranking on top.

## Setup

Requires Python 3.12+.

```bash
python -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

First run downloads model weights (~3.4 GB total) into `~/.cache/huggingface/`:
- `google/flan-t5-large` (~3 GB)
- `BAAI/bge-reranker-base` (~280 MB)
- `sentence-transformers/all-MiniLM-L6-v2` (~80 MB)

## Run

```bash
.venv/bin/streamlit run app.py --server.port 8501
```

Open `http://localhost:8501`, upload your 10-K PDFs in the sidebar, then ask questions in the chat.

## Tests

Plain-Python test suite (no pytest dependency):

```bash
bash tests/run_all.sh   # runs all four layers, fastest first
```

Or run individual layers:

| Layer | File | Speed | What it covers |
|---|---|---|---|
| Guardrails | `tests/test_guardrails.py` | < 1 s | input validation, confidence thresholds |
| Preprocessor | `tests/test_preprocessor.py` | < 1 s | chunker — decimals/abbreviations preserved, real overlap, size budget |
| Retriever | `tests/test_retriever.py` | ~10-30 s | embedding score range, BGE reranker demotes the historical "tax-on-revenue" trap, sigmoid calibration, BM25 sanity |
| End-to-end | `tests/test_e2e.py` | 2-4 min | 7-query battery against real index; needs `faiss_index` + `chunks.pkl` built (upload PDFs in the UI first) |

## System requirements

- 12 GB RAM available to the runtime (WSL2 users: set `[wsl2] memory=12GB` in `%USERPROFILE%\.wslconfig` and `wsl --shutdown`; default 50%-of-host cap may OOM on smaller hosts)
- ~5 GB free disk for model cache + index
- CPU-only (no CUDA required)

## Architecture overview

```
PDF ─→ pdfplumber ─→ clean_text ─→ sentence-aware chunks
                                          │
                                          ▼
                              ┌──────── chunks.pkl (cached)
                              ▼
                      embed (MiniLM-L6) ─→ FAISS IndexFlatIP ─→ faiss_index (cached)
                              │
                              │   query
                              ▼
              ┌─── hybrid retrieval (FAISS + BM25) ────┐
              │                                          │
              ▼                                          ▼
       BGE reranker ──→ top-k chunks ──→ flan-t5-large ──→ answer
              │                              │
              └──→ confidence score          └──→ source list
```

## Project layout

```
.
├── app.py                       # Streamlit UI + guardrails
├── rag_implementation.py        # RAGSystem, AdvancedRetrieval, SimpleBM25
├── financial_preprocessor.py    # PDF parsing + sentence-aware chunking
├── advanced_rag.py              # Adaptive retrieval helpers
├── fiximports.py                # PyTorch / asyncio compatibility shims
├── stylesheet.css               # Custom Streamlit theming
├── requirements.txt
├── fin_stmts/                   # Source 10-K PDFs
├── tests/                       # Plain-Python test suite
└── CLAUDE.md                    # Architecture notes for AI assistants
```
