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
from typing import Dict, Any, Tuple, List
import tempfile

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
    .file-uploader {
        padding: 1rem;
        border: 2px dashed #ccc;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .uploaded-file {
        padding: 0.5rem;
        background-color: #f0f8ff;
        border-radius: 0.3rem;
        margin-bottom: 0.3rem;
        border-left: 3px solid #4a6fa5;
    }
    .progress-section {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def process_uploaded_files(uploaded_files: List) -> List[str]:
    """
    Process uploaded files and save them to a temporary directory
    
    Args:
        uploaded_files: List of uploaded file objects from st.file_uploader
        
    Returns:
        List of file paths to the saved temporary files
    """
    temp_dir = tempfile.mkdtemp()
    file_paths = []
    
    for uploaded_file in uploaded_files:
        # Create a temporary file path
        file_path = os.path.join(temp_dir, uploaded_file.name)
        
        # Save uploaded file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        file_paths.append(file_path)
        
    return file_paths

def initialize_rag_system() -> RAGSystem:
    """
    Initialize the RAG system without building the index
    
    Returns:
        RAGSystem: The initialized RAG system without an index
    """
    # Initialize RAG system
    rag = RAGSystem(
        embedding_model_name='all-MiniLM-L6-v2',  # Small sentence transformer
        llm_model_name='google/flan-t5-base',     # Small open-source LLM
        index_path='faiss_index', 
        chunks_path='chunks.pkl'
    )
    
    return rag

def build_index_for_rag(rag: RAGSystem, file_paths: List[str]) -> Tuple[bool, str]:
    """
    Build the index for the RAG system using the provided file paths
    
    Args:
        rag: The RAG system
        file_paths: List of file paths to process
        
    Returns:
        Tuple of (success, message)
    """
    try:
        # Process financial statements
        preprocessor = FinancialDataPreprocessor()
        
        # Process documents and build index
        chunks, _ = preprocessor.process_financial_statements(file_paths)
        rag.build_index(chunks)
        
        return True, f"Index built successfully with {len(chunks)} chunks from {len(file_paths)} files!"
    except Exception as e:
        return False, f"Error building index: {str(e)}"

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

def display_uploaded_files(files: List):
    """Display the list of uploaded files with file info"""
    st.markdown("### Uploaded Financial Statements:")
    
    if not files:
        st.info("No files uploaded yet. Please upload financial statements to begin.")
        return
    
    for file in files:
        # Get file size in KB or MB
        file_size = file.size / 1024
        size_unit = "KB"
        
        if file_size > 1024:
            file_size /= 1024
            size_unit = "MB"
        
        # Display file info
        st.markdown(
            f"""<div class="uploaded-file">
                <strong>{file.name}</strong> ({file_size:.1f} {size_unit})
            </div>""", 
            unsafe_allow_html=True
        )

def main():
    # Page header
    st.title("💼 Financial RAG Chatbot by CAI Group 07")
    st.markdown("Upload financial statements and ask questions about the company's performance.")
    
    # Initialize session state for tracking uploads and processing
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = initialize_rag_system()
    
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    
    if "is_index_built" not in st.session_state:
        st.session_state.is_index_built = False
    
    if "processing_status" not in st.session_state:
        st.session_state.processing_status = ""
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Create a tab layout for upload/chat sections
    tabs = st.tabs(["📤 Upload Statements", "💬 Chat"])
    
    # Upload tab
    with tabs[0]:
        st.header("Upload Financial Statements")
        st.markdown("Upload financial statements in PDF, HTML, or DOCX format.")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload financial statements",
            type=["pdf", "html", "docx"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="file_uploader"
        )
        
        # Display uploaded files
        display_uploaded_files(uploaded_files)
        
        # Process files button
        if uploaded_files:
            if st.button("Process Files"):
                with st.spinner("Processing financial statements..."):
                    # Save uploaded files to temporary directory
                    file_paths = process_uploaded_files(uploaded_files)
                    
                    # Build index
                    success, message = build_index_for_rag(st.session_state.rag_system, file_paths)
                    
                    # Update session state
                    st.session_state.is_index_built = success
                    st.session_state.processing_status = message
                    st.session_state.uploaded_files = uploaded_files
                    
                    # Show result
                    if success:
                        st.success(message)
                        st.markdown("✅ **Ready to chat!** Switch to the Chat tab to ask questions about these financial statements.")
                    else:
                        st.error(message)
        
        # Show status if available
        if st.session_state.processing_status and not uploaded_files:
            if st.session_state.is_index_built:
                st.success(st.session_state.processing_status)
            else:
                st.error(st.session_state.processing_status)
        
        # Instructions
        with st.expander("File Requirements", expanded=False):
            st.markdown("""
            ### Supported File Types
            - **PDF**: Annual reports, 10-K, 10-Q filings
            - **HTML**: Financial statement web pages
            - **DOCX**: Financial reports in Word format
            
            ### Tips for Best Results
            - Use official financial statements from company websites or SEC filings
            - Files should contain clear financial data tables and text
            - For optimal results, include statements from at least 2 consecutive years
            - Larger files may take longer to process
            """)
    
    # Chat tab
    with tabs[1]:
        st.header("Ask Financial Questions")
        
        # Check if index is built
        if not st.session_state.is_index_built:
            st.warning("⚠️ Please upload and process financial statements before chatting.")
            return
        
        # Chat container
        chat_container = st.container()
        
        # Advanced options in sidebar
        st.sidebar.title("Options")
        show_chunks = st.sidebar.checkbox("Show source information", value=False)
        k_value = st.sidebar.slider("Number of sources to retrieve", min_value=1, max_value=5, value=2)
        
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
            st.rerun()
        
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
                        result = st.session_state.rag_system.query(query, k=k_value)
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
    
    # Instructions in sidebar
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
        financial questions based on the company financial statements you upload.
        
        **Features:**
        - Upload and analyze any company's financial statements
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