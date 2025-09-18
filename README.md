**RAG Chatbot**

Develop a **Retrieval-Augmented Generation (RAG)** model to answer financial questions based on company financial statements (last two years).

**IMPORTANT**

*   **Use Only Open-Source Embedding Models**
*   **Use a Small Open-Source Language Model (SLM) for Response Generation** (No proprietary APIs)
*   **Implement One Guardrail** (Either Input-side or Output-side)
*   **Develop an Application Interface** (Web-based)

**Advanced RAG Technique to be implemented**

1.  Chunk Merging & Adaptive Retrieval

**Tasks**

| Component | Details |
| --- | --- |
| 1. Data Collection & Preprocessing | Use the last two years of financials of a company. Clean and structure the data for retrieval. |
| 2. Basic RAG Implementation | Implement a simple RAG model:- Convert financial documents into text chunks.- Embed using a pre-trained model- Store and retrieve using a basic vector database |
| 3. UI Development using streamlit | Build an interactive UI:- Accept user queries.- Display answer & confidence score.- Ensure clear formatting & responsiveness. |
| 4. Advanced RAG Implementation | Improve retrieval by:- Using BM25 for keyword-based search alongside embeddings.- Testing different chunk sizes & retrieval methods for better accuracy.- Implementing re-ranking. |
| 5. Guard Rail Implementation | Implement one guardrail:- Input-Side: Validate and filter user queries to prevent irrelevant/harmful inputs.- Output-Side: Filter responses to remove hallucinated or misleading answers. |
| 6. Testing & Validation | Ask 3 test questions:- A relevant financial question (high-confidence).- A relevant financial question (low-confidence).- An irrelevant question (e.g., "What is the capital of France?") to check system robustness. |
