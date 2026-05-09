# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Financial RAG (Retrieval-Augmented Generation) Chatbot that answers questions based on company financial statements. It uses a hybrid retrieval pipeline (FAISS dense + BM25 sparse), a cross-encoder reranker for confidence calibration, and an instruction-tuned T5 model for answer synthesis. All models are open-source.

## Key Components and Architecture

### Core Modules

1. **app.py** — Streamlit web application interface
   - File upload, chat UI, session state, custom CSS
   - Input guardrail in `validate_query` (financial-relevance keyword check)
   - Output guardrail in `format_confidence` (High / Medium / Low labels by score thresholds)
   - Instantiates `RAGSystem(llm_model_name='google/flan-t5-large', enable_advanced_retrieval=...)`

2. **rag_implementation.py** — Core RAG system
   - `RAGSystem` — orchestrates embedding, retrieval, reranking, generation
     - Embedding model: `sentence-transformers/all-MiniLM-L6-v2`, L2-normalized
     - Vector store: FAISS `IndexFlatIP` (inner product on normalized vectors = cosine similarity)
     - LLM: `google/flan-t5-large` (770M params)
     - Cross-encoder: `BAAI/bge-reranker-base`, pinned to CPU (WSL2 GPU is too small)
     - `_rerank_with_cross_encoder` always runs on top-k candidates; sigmoid-calibrated [0,1] scores
   - `AdvancedRetrieval` — BM25 sparse retrieval, hybrid weighting, chunk merging, adaptive k/alpha selection
   - `SimpleBM25` — minimal in-memory BM25 implementation (no external dep)

3. **financial_preprocessor.py** — Document processing and chunking
   - `FinancialDataPreprocessor` — handles PDF, HTML, DOCX
   - PDF parsing routes through `_read_pdf_with_pdfplumber` (preserves word boundaries; PyPDF2's `_read_pdf` is kept as a fallback)
   - Sentence-aware chunking: protects decimals (`$309.1`) and common abbreviations (`Inc.`, `U.S.`, `e.g.`) with `<DOT>` placeholders before splitting on `(?<=[.!?])\s+`
   - Real (non-discardable) overlap: trailing sentences carried into the next chunk until cumulative length ≥ `chunk_overlap`

4. **advanced_rag.py** — Advanced RAG techniques (chunk merging, adaptive retrieval helpers)

5. **fiximports.py** — Import compatibility shims for PyTorch + async

### Tests

The `tests/` directory contains a plain-Python test suite (no pytest dependency):
- `test_guardrails.py` — input validation + confidence threshold mapping (~1 s)
- `test_preprocessor.py` — chunker regressions (decimals preserved, abbreviations preserved, overlap real, size budget) (~1 s)
- `test_retriever.py` — embedding score range, BGE reranker demotes the historical "tax-provision-on-revenue-query" trap, sigmoid calibration, BM25 sanity (~10-30 s, loads embedding + cross-encoder)
- `test_e2e.py` — 7-query battery against the real index (high-conf, low-conf, irrelevant categories) with peak-RSS sanity check (~2-4 minutes, loads flan-t5-large)
- `sample_data.py` — shared fixtures
- `run_all.sh` — runs all four in fastest-first order

Run a single layer with `.venv/bin/python tests/test_<layer>.py`. The retriever test stubs out the LLM via monkeypatch to avoid loading flan-t5-large for retrieval-only checks.

### Data Directory
- **fin_stmts/** — Google's 10-K financial statements

## Development Commands

### Setup
```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### Run the app
```bash
.venv/bin/streamlit run app.py --server.port 8501 --server.headless true
```

First model load downloads `flan-t5-large` (~3 GB) and `BAAI/bge-reranker-base` (~280 MB) into `~/.cache/huggingface/`. Subsequent runs are cached.

### Run tests
```bash
bash tests/run_all.sh          # everything, fastest first
.venv/bin/python tests/test_guardrails.py   # individual layer
```

## Implementation Details

### RAG Pipeline Flow
1. **Document parsing** — PDFs parsed with pdfplumber, cleaned (currency symbols stripped, whitespace normalized)
2. **Chunking** — sentence-aware splitting with abbreviation/decimal protection; overlap via trailing sentences
3. **Embedding** — `all-MiniLM-L6-v2`, L2-normalized
4. **Retrieval** — FAISS dense search (cosine similarity via inner product) over-fetches `max(k*2, 10)` candidates; optional BM25 hybrid via `AdvancedRetrieval.hybrid_retrieval`
5. **Reranking** — BGE cross-encoder always reruns on candidates; sigmoid maps logits to [0,1]
6. **Generation** — flan-t5-large with `num_beams=2`, `early_stopping=True`, `max_new_tokens=256`; instruction-style prompt with explicit "if not in context, say so" clause

### Confidence Score
Top-1 cross-encoder score (no method/domain bonuses). Honest [0,1] calibration — should NOT saturate at 1.0 on imperfect matches. Thresholds: High ≥ 0.7, Medium ≥ 0.4, Low otherwise.

### Key Parameters
- Default chunk size: 500 chars; overlap: 50 chars (settable in `FinancialDataPreprocessor.__init__`)
- Top-k retrieval: 5 (configurable via UI slider)
- LLM: `num_beams=2`, `max_new_tokens=256`, `min_length=30`, `no_repeat_ngram_size=3`, `early_stopping=True`
- Cross-encoder: `device='cpu'` (forced — WSL2's 2 GiB integrated GPU is too small for BGE)

### Guardrails
- **Input** (`validate_query` in app.py): rejects empty / too-short / non-financial queries via keyword check
- **Output** (`format_confidence` in app.py): three-band confidence label informs the user when answers are low-confidence

## Memory Profile

The full pipeline peaks at ~5-9 GB RSS during inference (T5-large beam search dominates). On WSL2 the default 50%-of-host memory cap can OOM-kill the process; on hosts with ≤ 16 GB RAM, set `[wsl2] memory=12GB` in `%USERPROFILE%\.wslconfig` and `wsl --shutdown` from PowerShell. The E2E test asserts peak RSS ≤ 9 GB as a regression sanity check.

## Testing Approach

Three categories per the assignment brief:
1. Relevant financial questions expected to score high confidence
2. Relevant financial questions expected to score low confidence
3. Irrelevant questions to test refusal

`tests/test_e2e.py` encodes a 7-query battery covering all three.

## Important Considerations

- All models are open-source (no proprietary APIs)
- CPU-only by design (no CUDA dependency in code paths)
- Caches FAISS index (`faiss_index`) and chunks (`chunks.pkl`) on disk after first build; both are `.gitignore`d (rebuild from PDFs in UI on a fresh checkout)
- Streamlit session state for chat history
