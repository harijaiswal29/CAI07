# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Financial RAG (Retrieval-Augmented Generation) Chatbot that answers questions based on company financial statements. It implements advanced RAG techniques including chunk merging, adaptive retrieval, and BM25 hybrid search alongside embeddings.

## Key Components and Architecture

### Core Modules

1. **app.py** - Main Streamlit web application interface
   - Handles UI/UX, file uploads, and user interactions
   - Implements input/output guardrails for query validation
   - Manages session state and chat history
   - Uses custom CSS styling for enhanced visual presentation

2. **rag_implementation.py** - Core RAG system implementation
   - `RAGSystem` class: Main orchestrator for the RAG pipeline
   - `AdvancedRetrieval` class: Implements BM25, hybrid search, and re-ranking
   - Uses SentenceTransformer for embeddings (all-MiniLM-L6-v2)
   - Uses T5-small model for response generation
   - Implements FAISS vector store for similarity search

3. **financial_preprocessor.py** - Document processing and chunking
   - `FinancialDataPreprocessor` class: Handles PDF, HTML, and DOCX parsing
   - Implements smart chunking with configurable size and overlap
   - Extracts and structures financial data from statements
   - Handles table extraction and formatting

4. **advanced_rag.py** - Advanced RAG techniques
   - Implements chunk merging strategies
   - Adaptive retrieval based on query complexity

5. **fiximports.py** - Import compatibility fixes for PyTorch and async operations

### Data Directory
- **fin_stmts/** - Contains Google's 10-K financial statements (2023, 2024)

## Development Commands

### Setup and Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Common Development Tasks
```bash
# Run the Streamlit app on a specific port
streamlit run app.py --server.port 8501

# Kill any running Streamlit processes
pkill -f streamlit

# Check if the app is running
ps aux | grep streamlit
```

## Implementation Details

### RAG Pipeline Flow
1. **Document Processing**: PDFs are loaded and chunked with overlap
2. **Embedding**: Chunks are embedded using SentenceTransformer
3. **Retrieval**: Hybrid approach using both semantic (FAISS) and keyword (BM25) search
4. **Re-ranking**: Retrieved chunks are re-ranked based on relevance scores
5. **Generation**: T5-small generates responses based on retrieved context
6. **Guardrails**: Input validation and output filtering for safety

### Key Parameters
- Default chunk size: 500 tokens
- Default chunk overlap: 50 tokens
- Top-k retrieval: 5-10 chunks (configurable)
- Confidence thresholds: High (>0.7), Medium (0.4-0.7), Low (<0.4)

### Guardrails Implementation
- **Input-side**: Validates financial relevance, filters harmful queries
- **Output-side**: Confidence scoring, hallucination detection

## Testing Approach

The system should be tested with three categories of questions:
1. Relevant financial questions expecting high confidence
2. Relevant financial questions expecting low confidence  
3. Irrelevant questions to test robustness

## Important Considerations

- All models are open-source (no proprietary APIs)
- Uses CPU-based FAISS for portability
- Implements caching for processed documents
- Session state management for chat history
- Progress indicators for long-running operations