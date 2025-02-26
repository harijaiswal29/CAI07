import sys
import os
import asyncio

# Fix for PyTorch import issues
try:
    # Attempt to fix the event loop issue
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Import the fix before any other imports
import fiximports

import streamlit as st
import re
from typing import Dict, Any, Tuple

# Add parent directory to path to import our modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import our RAG system and preprocessor
from rag_implementation import RAGSystem
from financial_preprocessor import FinancialDataPreprocessor

# Set page config
st.set_page_config(
    page_title="Financial RAG Chatbot by CAI Group 07",
    page_icon="💰",
    layout="wide",
)

# Apply custom CSS for better formatting
st.markdown("""
<style>
    .confidence-high { color: green; font-weight: bold; }
    .confidence-medium { color: orange; font-weight: bold; }
    .confidence-low { color: red; font-weight: bold; }
    .stButton button { width: 100%; }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .chat-message.user {
        background-color: #f0f2f6;
    }
    .chat-message.assistant {
        background-color: #e6f3ff;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_rag_system() -> RAGSystem:
    """
    Load or initialize the RAG system
    
    Returns:
        RAGSystem: The initialized RAG system
    """
    # Check if index files exist
    index_path = 'faiss_index'
    chunks_path = 'chunks.pkl'
    
    # Initialize RAG system
    rag = RAGSystem(
        embedding_model_name='all-MiniLM-L6-v2',  # Small sentence transformer
        llm_model_name='google/flan-t5-base',     # Small open-source LLM
        index_path=index_path, 
        chunks_path=chunks_path
    )
    
    # If files don't exist, build the index
    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        st.info("First-time setup: Building search index. This may take a few minutes...")
        
        # Process financial statements
        preprocessor = FinancialDataPreprocessor()
        
        # Replace with actual file paths
        #file_paths = ['fin_stmts/goog-10-k-2024.pdf']  # Example path, replace with your file paths
        file_paths = [
                'fin_stmts/goog-10-k-2024.pdf',  # 2024 financial statement
                'fin_stmts/goog-10-k-2023.pdf'   # 2023 financial statement
            ]
        
        try:
            # Process documents and build index
            chunks, _ = preprocessor.process_financial_statements(file_paths)
            rag.build_index(chunks)
            st.success("Index built successfully!")
        except Exception as e:
            st.error(f"Error building index: {str(e)}")
    
    return rag

def validate_query(query: str) -> Dict[str, Any]:
    """
    Input-side guardrail: Validate and filter user queries
    to prevent irrelevant/harmful inputs
    
    Args:
        query: User query text
        
    Returns:
        Dictionary with 'valid' (bool) and 'message' (str) fields
    """
    # Check for empty queries
    if not query or query.strip() == "":
        return {"valid": False, "message": "Please enter a question."}
    
    # Check query length
    if len(query) < 5:
        return {"valid": False, "message": "Your question is too short. Please be more specific."}
    
    # Check for financial relevance using keywords
    financial_keywords = [
        'revenue', 'profit', 'loss', 'income', 'expense', 'asset', 'liability',
        'balance', 'sheet', 'statement', 'cash', 'flow', 'equity', 'dividend',
        'earnings', 'quarter', 'annual', 'fiscal', 'year', 'ratio', 'margin',
        'growth', 'decline', 'increase', 'decrease', 'financial', 'company'
    ]
    
    # Check if query contains at least one financial keyword
    if not any(keyword in query.lower() for keyword in financial_keywords):
        return {
            "valid": False, 
            "message": "Please ask a question related to financial statements or company performance."
        }
    
    return {"valid": True, "message": ""}

def format_confidence(score: float) -> Tuple[str, str]:
    """
    Format confidence score with appropriate color and label
    
    Args:
        score: Confidence score (0-1)
        
    Returns:
        Tuple of (CSS class, label)
    """
    if score >= 0.7:
        return "confidence-high", "High"
    elif score >= 0.4:
        return "confidence-medium", "Medium"
    else:
        return "confidence-low", "Low"

def display_chat_message(role: str, content: str, confidence: float = None, chunks: list = None, show_chunks: bool = False):
    """Display a formatted chat message"""
    message_class = "user" if role == "user" else "assistant"
    
    with st.container():
        st.markdown(f'<div class="chat-message {message_class}">', unsafe_allow_html=True)
        
        # Avatar and name
        if role == "user":
            st.markdown("### 👤 You:")
        else:
            st.markdown("### 🤖 Assistant:")
        
        # Message content
        st.markdown(content)
        
        # Confidence score for assistant messages
        if role == "assistant" and confidence is not None:
            css_class, confidence_label = format_confidence(confidence)
            st.markdown(
                f"**Confidence Score:** <span class='{css_class}'>{confidence_label} ({confidence:.2f})</span>", 
                unsafe_allow_html=True
            )
            
            # Show chunks if enabled
            if show_chunks and chunks:
                with st.expander("View Source Information", expanded=False):
                    for i, chunk in enumerate(chunks):
                        st.markdown(f"**Source {i+1}** (Relevance: {chunk['score']:.3f})")
                        st.text(chunk['chunk'][:300] + "..." if len(chunk['chunk']) > 300 else chunk['chunk'])
        
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    # Page header
    st.title("💼 Financial RAG Chatbot by CAI Group 07")
    st.markdown("Ask questions about the company's financial statements from the last two years.")
    
    # Load RAG system
    with st.spinner("Loading RAG system..."):
        rag = load_rag_system()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Chat container
    chat_container = st.container()
    
    # Advanced options in sidebar
    st.sidebar.title("Options")
    show_chunks = st.sidebar.checkbox("Show source information", value=False)
    k_value = st.sidebar.slider("Number of sources to retrieve", min_value=1, max_value=2, value=2)
    
    # Display chat history
    with chat_container:
        for message in st.session_state.messages:
            display_chat_message(
                role=message["role"],
                content=message["content"],
                confidence=message.get("confidence"),
                chunks=message.get("chunks"),
                show_chunks=show_chunks
            )
    
    # Input area
    st.markdown("### Ask a financial question:")
    query = st.text_input(
        label="Your question", 
        key="query_input",
        placeholder="Example: What was the total revenue for 2023?",
        label_visibility="collapsed"
    )
    
    # Action buttons
    col1, col2 = st.columns([4, 1])
    with col1:
        submit_button = st.button("Submit Question", use_container_width=True)
    with col2:
        clear_button = st.button("Clear Chat", use_container_width=True)
    
    if clear_button:
        st.session_state.messages = []
        st.experimental_rerun()
    
    # Process the query
    if submit_button and query:
        # Apply input guardrail
        validation = validate_query(query)
        
        if not validation["valid"]:
            st.error(validation["message"])
        else:
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": query})
            
            # Display the user message immediately
            with chat_container:
                display_chat_message(role="user", content=query)
            
            # Show a spinner while processing
            with st.spinner("Generating response..."):
                try:
                    # Get response from RAG system
                    result = rag.query(query, k=k_value)
                    response = result["response"]
                    confidence = result["confidence_score"]
                    retrieved_chunks = result["retrieved_chunks"]
                    
                    # Add bot response to chat
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response, 
                        "confidence": confidence,
                        "chunks": retrieved_chunks
                    })
                    
                    # Display the assistant message
                    with chat_container:
                        display_chat_message(
                            role="assistant", 
                            content=response, 
                            confidence=confidence,
                            chunks=retrieved_chunks,
                            show_chunks=show_chunks
                        )
                    
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
    
    # Instructions
    with st.sidebar.expander("Example Questions", expanded=True):
        st.markdown("""
        - What was the total revenue for 2023?
        - How much did operating expenses increase from 2022 to 2023?
        - What is the company's debt-to-equity ratio?
        - What are the main factors affecting profitability?
        - How has the cash position changed year-over-year?
        """)
    
    # About section
    with st.sidebar.expander("About", expanded=False):
        st.markdown("""
        ### Financial RAG Chatbot
        
        This application uses Retrieval-Augmented Generation (RAG) to provide accurate answers to 
        financial questions based on the last two years of company financial statements.
        
        **Features:**
        - Query financial data with natural language
        - View confidence scores for each response
        - Examine source information for transparency
        - Input validation to ensure relevant questions
        
        **Technical Details:**
        - Embedding: all-MiniLM-L6-v2
        - Language Model: google/flan-t5-base
        - Guardrail: Input-side validation
        """)

if __name__ == "__main__":
    main()
