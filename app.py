import sys
import os
import asyncio

print(f"Current working directory: {os.getcwd()}")
print(f"Files in directory: {os.listdir('.')}")
# Fix for PyTorch import issues
try:
    # Attempt to fix the event loop issue
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Import the fix before any other imports
try:
    import fiximports
except ImportError:
    pass

import streamlit as st
import re
from typing import Dict, Any, Tuple, List
import tempfile

# Add parent directory to path to import our modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import our RAG system and preprocessor
try:
    from rag_implementation import RAGSystem, AdvancedRetrieval
    from financial_preprocessor import FinancialDataPreprocessor
except ImportError as e:
    st.error(f"Error importing required modules: {str(e)}")
    st.stop()

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
    .method-tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.8rem;
        margin-left: 5px;
    }
    .method-embedding { background-color: #e3f2fd; }
    .method-bm25 { background-color: #e8f5e9; }
    .method-hybrid { background-color: #fff8e1; }
    .method-merged { background-color: #f3e5f5; }
    .method-reranked { background-color: #ffebee; }
    .adv-param {
        padding: 5px;
        border-radius: 4px;
        background-color: #f5f5f5;
        margin-bottom: 4px;
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
    # Initialize RAG system with advanced retrieval
    try:
        print("Initializing RAG system...")
        rag = RAGSystem(
            embedding_model_name='all-MiniLM-L6-v2',  # Small sentence transformer
            llm_model_name='google/flan-t5-base',     # Small open-source LLM
            index_path='faiss_index', 
            chunks_path='chunks.pkl',
            enable_advanced_retrieval=False  # Default to false, will be toggled by UI
        )
        print("RAG system initialized successfully")
        return rag
    except Exception as e:
        print(f"ERROR initializing RAG system: {str(e)}")
        st.error(f"Error initializing RAG system: {str(e)}")
        st.stop()

        

def build_index_for_rag(rag: RAGSystem, file_paths: List[str], chunk_size: int = 500, chunk_overlap: int = 50) -> Tuple[bool, str]:
    """
    Build the index for the RAG system using the provided file paths
    
    Args:
        rag: The RAG system
        file_paths: List of file paths to process
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        Tuple of (success, message)
    """
    try:
        # Process financial statements with specified chunk size and overlap
        preprocessor = FinancialDataPreprocessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
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

# In app.py, enhance the format_confidence function
def format_confidence(score: float) -> Tuple[str, str]:
    """Format confidence score with appropriate color and label"""
    if score >= 0.7:
        return "confidence-high", "High (Highly relevant sources found)"
    elif score >= 0.4:
        return "confidence-medium", "Medium (Somewhat relevant sources found)"
    else:
        return "confidence-low", "Low (Limited relevant information found)"

def format_method_tag(method: str) -> str:
    """
    Format retrieval method as a colored tag
    
    Args:
        method: The retrieval method name
        
    Returns:
        HTML string for the method tag
    """
    method_class = "method-embedding"
    
    if "bm25" in method:
        method_class = "method-bm25"
    elif "hybrid" in method:
        method_class = "method-hybrid"
    elif "merged" in method:
        method_class = "method-merged"
    elif "reranked" in method:
        method_class = "method-reranked"
    
    return f'<span class="method-tag {method_class}">{method}</span>'

def display_chat_message(role: str, content: str, confidence: float = None, chunks: list = None, 
                         show_chunks: bool = False, params: dict = None, show_params: bool = False):
    """Display a formatted chat message with advanced features"""
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
        
        # Confidence score and advanced features for assistant messages
        if role == "assistant" and confidence is not None:
            css_class, confidence_label = format_confidence(confidence)
            st.markdown(
                f"**Confidence Score:** <span class='{css_class}'>{confidence_label} ({confidence:.2f})</span>", 
                unsafe_allow_html=True
            )
            
            # Show retrieval parameters if available
            if show_params and params:
                with st.expander("Retrieval Parameters", expanded=False):
                    for key, value in params.items():
                        if key == 'k':
                            st.markdown(f"<div class='adv-param'>Sources retrieved: {value}</div>", unsafe_allow_html=True)
                        elif key == 'alpha':
                            st.markdown(f"<div class='adv-param'>Embedding weight: {value:.2f} (BM25 weight: {1-value:.2f})</div>", unsafe_allow_html=True)
                        elif key == 'merged' and value:
                            st.markdown(f"<div class='adv-param'>Chunk merging: Enabled</div>", unsafe_allow_html=True)
                        elif key == 'reranked' and value:
                            st.markdown(f"<div class='adv-param'>Cross-encoder reranking: Enabled</div>", unsafe_allow_html=True)
            
            # Show chunks if enabled
            if show_chunks and chunks:
                with st.expander("View Source Information", expanded=False):
                    for i, chunk in enumerate(chunks):
                        # Format method tag if available
                        method_info = ""
                        if 'method' in chunk:
                            method_info = format_method_tag(chunk['method'])
                        
                        st.markdown(f"**Source {i+1}** (Relevance: {chunk['score']:.3f}) {method_info}", unsafe_allow_html=True)
                        
                        # Show merged count if available
                        if chunk.get('merged_count', 0) > 1:
                            st.markdown(f"*Merged from {chunk['merged_count']} chunks*")
                            
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
        
        # Advanced chunking options
        with st.expander("Advanced Chunking Options", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                chunk_size = st.slider("Chunk Size", 200, 1000, 500, 100, 
                                      help="Size of text chunks in characters")
            with col2:
                chunk_overlap = st.slider("Chunk Overlap", 0, 200, 50, 10, 
                                         help="Overlap between chunks in characters")
        
        # Display uploaded files
        display_uploaded_files(uploaded_files)
        
        # Process files button
        if uploaded_files:
            if st.button("Process Files"):
                with st.spinner("Processing financial statements..."):
                    # Save uploaded files to temporary directory
                    file_paths = process_uploaded_files(uploaded_files)
                    
                    # Build index with specified chunk size and overlap
                    success, message = build_index_for_rag(
                        st.session_state.rag_system, 
                        file_paths,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    
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
        st.sidebar.title("RAG Options")
        show_chunks = st.sidebar.checkbox("Show source information", value=False)
        show_params = st.sidebar.checkbox("Show retrieval parameters", value=False)
        k_value = st.sidebar.slider("Number of sources to retrieve", min_value=1, max_value=7, value=5)
        



        # Advanced RAG options - simplified for basic implementation
        use_advanced = st.sidebar.checkbox("Enable Advanced RAG", value=False, 
                                help="Use advanced retrieval techniques")
        
        # Apply settings to RAG system
        rag_system = st.session_state.rag_system
        #rag_system.enable_advanced_retrieval = use_advanced
        
        if rag_system:
            # Log the before state
            print(f"Before toggle: Advanced RAG enabled = {rag_system.enable_advanced_retrieval}")
            
            # Set the advanced option
            rag_system.enable_advanced_retrieval = use_advanced
            
            # Log the after state and verify attributes
            print(f"After toggle: Advanced RAG enabled = {rag_system.enable_advanced_retrieval}")
            print(f"Advanced retrieval available: {hasattr(rag_system, 'advanced_retrieval')}")
            
            # If advanced is enabled but not initialized, initialize it now
            if use_advanced and not hasattr(rag_system, 'advanced_retrieval'):
                try:
                    print("Initializing advanced retrieval...")
                    # This will use the embedded AdvancedRetrieval in rag_implementation.py
                    rag_system.advanced_retrieval = AdvancedRetrieval()  
                    print("Advanced retrieval initialized successfully")
                except Exception as e:
                    print(f"ERROR initializing advanced retrieval: {str(e)}")
                    st.sidebar.warning(f"Could not initialize advanced retrieval: {str(e)}")
        


        # Display chat history
        with chat_container:
            for message in st.session_state.messages:
                display_chat_message(
                    role=message["role"],
                    content=message["content"],
                    confidence=message.get("confidence"),
                    chunks=message.get("chunks"),
                    show_chunks=show_chunks,
                    params=message.get("params"),
                    show_params=show_params
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
                        print(f"Advanced RAG enabled: {rag_system.enable_advanced_retrieval}")

                        # Get response from RAG system
                        result = rag_system.query(query, k=k_value)

                        print(f"Using method: {'Advanced' if 'retrieval_params' in result else 'Basic'}")
                        if 'retrieval_params' in result:
                            print(f"Parameters used: {result['retrieval_params']}")


                        response = result["response"]
                        confidence = result["confidence_score"]
                        retrieved_chunks = result["retrieved_chunks"]
                        retrieval_params = result.get("retrieval_params", None)
                        
                        # Add bot response to chat
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response, 
                            "confidence": confidence,
                            "chunks": retrieved_chunks,
                            "params": retrieval_params
                        })
                        
                        # Display the assistant message
                        with chat_container:
                            display_chat_message(
                                role="assistant", 
                                content=response, 
                                confidence=confidence,
                                chunks=retrieved_chunks,
                                show_chunks=show_chunks,
                                params=retrieval_params,
                                show_params=show_params
                            )
                        
                    except Exception as e:
                        st.error(f"Error processing query: {str(e)}")
    
    # Instructions in sidebar
    with st.sidebar.expander("Example Questions", expanded=True):
        st.markdown("""
        - What was the total revenue for 2023?
        - How much did operating expenses increase from 2022 to 2023?        
        - What are the main factors affecting profitability?
        - How has the cash position changed year-over-year?
        """)
    
    # About section
    with st.sidebar.expander("About", expanded=False):
        st.markdown("""
        ### Financial RAG Chatbot
        
        This application by CAI Group 07 uses Retrieval-Augmented Generation (RAG) to provide accurate answers to 
        financial questions based on the company financial statements uploaded.
        
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