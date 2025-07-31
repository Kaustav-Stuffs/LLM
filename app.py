# import streamlit as st
# import os
# import json
# import tempfile
# import logging
# from typing import List, Dict, Optional
# import time

# from utils.device_detector import get_optimal_device
# from utils.document_processor import DocumentProcessor
# from utils.hybrid_search import HybridSearchRAG
# from utils.llm_handler import LLMHandler
# from utils.query_processor import QueryProcessor

# # Configure logging
# # Replace the logging configuration at the top of app.py with:
# import logging
# import warnings

# # Suppress specific warnings
# warnings.filterwarnings("ignore", category=UserWarning, module="streamlit")
# warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
# warnings.filterwarnings("ignore", category=DeprecationWarning)

# # Configure logging with more specific levels
# logging.basicConfig(
#     level=logging.WARNING,  # Changed from INFO to WARNING
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.StreamHandler()
#     ]
# )

# # Set specific loggers to WARNING or ERROR
# logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
# logging.getLogger("transformers").setLevel(logging.WARNING)
# logging.getLogger("torch").setLevel(logging.WARNING)
# logging.getLogger("urllib3").setLevel(logging.WARNING)

# logger = logging.getLogger(__name__)


# # Page configuration
# st.set_page_config(
#     page_title="General-Purpose RAG Chatbot",
#     page_icon="🤖",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Initialize session state
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# if "documents" not in st.session_state:
#     st.session_state.documents = []

# if "search_engine" not in st.session_state:
#     st.session_state.search_engine = None

# if "document_processor" not in st.session_state:
#     device = get_optimal_device()
#     st.session_state.document_processor = DocumentProcessor(device=device)

# if "llm_handler" not in st.session_state:
#     st.session_state.llm_handler = LLMHandler()

# if "query_processor" not in st.session_state:
#     st.session_state.query_processor = QueryProcessor()

# # Main title and description
# st.title("🤖 General-Purpose RAG Chatbot")
# st.markdown("""
# Upload your documents and chat with them! This chatbot provides answers based **only** on your uploaded content.
# Supports PDF, DOCX, TXT, JSON, and CSV files.
# """)

# # Sidebar for configuration
# with st.sidebar:
#     st.header("⚙️ Configuration")
    
#     # Device information
#     device = get_optimal_device()
#     device_icon = "🚀" if device == "cuda" else "🔥" if device == "mps" else "💻"
#     st.info(f"{device_icon} Using device: **{device.upper()}**")
    
#     # Model selection
#     st.subheader("🧠 LLM Selection")
#     model_type = st.radio(
#         "Choose model type:",
#         ["ONLINE (Gemini)", "OFFLINE (Ollama)"],
#         help="Online models require internet connection. Offline models run locally."
#     )
    
#     if model_type == "ONLINE (Gemini)":
#         gemini_model = st.selectbox(
#             "Gemini Model:",
#             ["gemini-2.0-flash", "gemini-2.5-pro"],
#             index=0,
#             help="Flash is faster, Pro is more capable"
#         )
#         api_key = os.getenv("GEMINI_API_KEY")
#         if not api_key:
#             st.warning("⚠️ GEMINI_API_KEY not found in environment variables")
#     else:
#         ollama_models = st.session_state.llm_handler.get_available_ollama_models()
#         if ollama_models:
#             ollama_model = st.selectbox(
#                 "Ollama Model:",
#                 ollama_models,
#                 help="Local models available on your system"
#             )
#         else:
#             st.error("❌ No Ollama models found. Please install Ollama and pull some models.")
#             ollama_model = None
    
#     # Search parameters
#     st.subheader("🔍 Search Settings")
#     vector_weight = st.slider("Vector Search Weight", 0.0, 1.0, 0.7, 0.1)
#     bm25_weight = 1.0 - vector_weight
#     st.write(f"BM25 Weight: {bm25_weight:.1f}")
    
#     top_k = st.slider("Number of Results", 1, 10, 3)
#     min_relevance = st.slider("Minimum Relevance Score", 0.0, 1.0, 0.5, 0.05)

# # File upload section
# st.header("📁 Document Upload")

# uploaded_files = st.file_uploader(
#     "Upload your documents",
#     type=["pdf", "docx", "txt", "json", "csv"],
#     accept_multiple_files=True,
#     help="Supported formats: PDF, DOCX, TXT, JSON, CSV"
# )

# if uploaded_files:
#     with st.expander("📊 Document Processing Status", expanded=True):
#         progress_bar = st.progress(0)
#         status_text = st.empty()
        
#         processed_docs = []
#         total_files = len(uploaded_files)
        
#         for i, uploaded_file in enumerate(uploaded_files):
#             status_text.text(f"Processing {uploaded_file.name}...")
            
#             try:
#                 # Save uploaded file temporarily
#                 with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
#                     tmp_file.write(uploaded_file.getvalue())
#                     tmp_path = tmp_file.name
                
#                 # Process the document
#                 docs = st.session_state.document_processor.process_file(tmp_path, uploaded_file.name)
#                 processed_docs.extend(docs)
                
#                 # Show processing details
#                 st.success(f"✅ {uploaded_file.name}: {len(docs)} chunks extracted")
                
#                 # Clean up temp file
#                 os.unlink(tmp_path)
                
#             except Exception as e:
#                 st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
#                 logger.error(f"Error processing {uploaded_file.name}: {e}")
            
#             progress_bar.progress((i + 1) / total_files)
        
#         if processed_docs:
#             status_text.text("Building search index...")
            
#             # Create search engine
#             try:
#                 st.session_state.search_engine = HybridSearchRAG(
#                     documents=processed_docs,
#                     vector_weight=vector_weight,
#                     bm25_weight=bm25_weight,
#                     top_k=top_k,
#                     device=device
#                 )
#                 st.session_state.documents = processed_docs
                
#                 st.success(f"🎉 Successfully indexed {len(processed_docs)} document chunks!")
                
#                 # Show document statistics
#                 col1, col2, col3 = st.columns(3)
#                 with col1:
#                     st.metric("Total Documents", len(uploaded_files))
#                 with col2:
#                     st.metric("Total Chunks", len(processed_docs))
#                 with col3:
#                     avg_length = sum(len(doc['text']) for doc in processed_docs) // len(processed_docs)
#                     st.metric("Avg Chunk Length", f"{avg_length} chars")
                
#             except Exception as e:
#                 st.error(f"❌ Error building search index: {str(e)}")
#                 logger.error(f"Error building search index: {e}")
        
#         status_text.text("✅ Processing complete!")

# # Chat interface
# st.header("💬 Chat Interface")

# # Display chat history
# for message in st.session_state.chat_history:
#     with st.chat_message(message["role"]):
#         st.write(message["content"])
#         if message["role"] == "assistant" and "sources" in message:
#             with st.expander("📚 Sources", expanded=False):
#                 for i, source in enumerate(message["sources"], 1):
#                     st.write(f"**Source {i}** (Score: {source['score']:.3f})")
#                     st.write(f"```\n{source['text'][:300]}{'...' if len(source['text']) > 300 else ''}\n```")

# # Chat input
# # Replace the existing chat processing section with:
# if prompt := st.chat_input("Ask a question about your documents..."):
#     if not st.session_state.search_engine:
#         st.error("Please upload and process documents first!")
#     else:
#         # Add user message to chat history BEFORE processing
#         user_message = {"role": "user", "content": prompt}
#         st.session_state.chat_history.append(user_message)
        
#         # Display user message
#         with st.chat_message("user"):
#             st.write(prompt)
        
#         # Process query and generate response
#         with st.chat_message("assistant"):
#             with st.spinner("Thinking..."):
#                 try:
#                     # Configure LLM
#                     if model_type == "ONLINE (Gemini)":
#                         if not st.session_state.llm_handler.configure_gemini(gemini_model):
#                             st.error("Failed to configure Gemini model!")
#                             st.stop()
#                         model_choice = "gemini"
#                     else:
#                         if ollama_model:
#                             if not st.session_state.llm_handler.configure_ollama(ollama_model):
#                                 st.error("Failed to configure Ollama model!")
#                                 st.stop()
#                             model_choice = "ollama"
#                         else:
#                             st.error("No Ollama model selected!")
#                             st.stop()
                    
#                     # Process the query
#                     result = st.session_state.query_processor.process_query(
#                         query=prompt,
#                         search_engine=st.session_state.search_engine,
#                         llm_handler=st.session_state.llm_handler,
#                         model_choice=model_choice,
#                         min_relevance_score=min_relevance
#                     )
                    
#                     # Display response
#                     response = result.get("answer", "Sorry, I couldn't generate a response.")
#                     st.write(response)
                    
#                     # Prepare sources for display and storage
#                     sources = []
#                     if "matched_documents" in result and "relevance_scores" in result:
#                         for doc, score in zip(result["matched_documents"], result["relevance_scores"]):
#                             sources.append({
#                                 "text": doc,
#                                 "score": score
#                             })
                    
#                     # Add assistant message to chat history with complete information
#                     assistant_message = {
#                         "role": "assistant",
#                         "content": response,
#                         "sources": sources,
#                         "timestamp": time.time()  # Add timestamp for better tracking
#                     }
#                     st.session_state.chat_history.append(assistant_message)
                    
#                     # Show sources
#                     if sources:
#                         with st.expander("📚 Sources", expanded=False):
#                             for i, source in enumerate(sources, 1):
#                                 st.write(f"**Source {i}** (Score: {source['score']:.3f})")
#                                 st.write(f"``````")
                
#                 except Exception as e:
#                     error_msg = f"Error processing your question: {str(e)}"
#                     st.error(error_msg)
#                     # Add error to chat history
#                     error_message = {
#                         "role": "assistant",
#                         "content": error_msg,
#                         "sources": [],
#                         "timestamp": time.time(),
#                         "error": True
#                     }
#                     st.session_state.chat_history.append(error_message)
#                     logger.error(f"Error in chat processing: {e}")


# # Clear chat button
# if st.session_state.chat_history:
#     if st.button("🗑️ Clear Chat History"):
#         st.session_state.chat_history = []
#         st.rerun()

# # Footer
# st.markdown("---")
# st.markdown("**Note**: This chatbot answers questions based **only** on your uploaded documents. It does not use external knowledge.")

import streamlit as st
import os
import tempfile
import logging
import time
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="streamlit")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

from utils.device_detector import get_optimal_device
from utils.document_processor import DocumentProcessor
from utils.hybrid_search import HybridSearchRAG
from utils.llm_handler import LLMHandler
from utils.query_processor import QueryProcessor

# Page configuration
st.set_page_config(
    page_title="General-Purpose RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "documents" not in st.session_state:
    st.session_state.documents = []
if "search_engine" not in st.session_state:
    st.session_state.search_engine = None
if "document_processor" not in st.session_state:
    device = get_optimal_device()
    st.session_state.document_processor = DocumentProcessor(device=device)
if "llm_handler" not in st.session_state:
    st.session_state.llm_handler = LLMHandler()
if "query_processor" not in st.session_state:
    st.session_state.query_processor = QueryProcessor()

# Main title and description
st.title("🤖 General-Purpose RAG Chatbot")
st.markdown("""
Upload your documents to have their content automatically structured into a question-answer format. Then, chat with them! 
This chatbot provides answers based **only** on your uploaded content. Supports PDF, DOCX, TXT, JSON, and CSV files.
""")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    device = get_optimal_device()
    device_icon = "🚀" if device == "cuda" else "🔥" if device == "mps" else "💻"
    st.info(f"{device_icon} Using device: **{device.upper()}**")
    
    st.subheader("🧠 LLM Selection")
    model_type = st.radio(
        "Choose model type (for answering and structuring):",
        ["ONLINE (Gemini)", "OFFLINE (Ollama)"],
        help="The selected model will be used for both structuring documents and answering questions."
    )
    
    if model_type == "ONLINE (Gemini)":
        gemini_model = st.selectbox(
            "Gemini Model:", ["gemini-2.5-flash", "gemini-2.5-pro"], index=0,
            help="Flash is faster, Pro is more capable"
        )
        api_key = os.getenv("GEMINI_API_KEY", "AIzaSyDxYogQnjYmTbvghsqSWCKtot5DpygC8hA")
        if not api_key:
            st.warning("⚠️ GEMINI_API_KEY not found in environment variables")
    else:
        ollama_models = st.session_state.llm_handler.get_available_ollama_models()
        if ollama_models:
            ollama_model = st.selectbox("Ollama Model:", ollama_models)
        else:
            st.error("❌ No Ollama models found. Please install Ollama and pull some models.")
            ollama_model = None
    
    st.subheader("🔍 Search Settings")
    vector_weight = st.slider("Vector Search Weight", 0.0, 1.0, 0.7, 0.1)
    bm25_weight = 1.0 - vector_weight
    st.write(f"BM25 Weight: {bm25_weight:.1f}")
    
    top_k = st.slider("Number of Results", 1, 10, 3)
    min_relevance = st.slider("Minimum Relevance Score", 0.0, 1.0, 0.5, 0.05)

# File upload section
st.header("📁 Document Upload")

uploaded_files = st.file_uploader(
    "Upload documents to structure and index them",
    type=["pdf", "docx", "txt", "json", "csv"],
    accept_multiple_files=True
)

if uploaded_files:
    with st.expander("📊 Document Processing Status", expanded=True):
        
        # Configure the LLM first, as it's needed for structuring
        llm_configured = False
        with st.spinner("Configuring LLM for document processing..."):
            if model_type == "ONLINE (Gemini)":
                if st.session_state.llm_handler.configure_gemini(gemini_model):
                    llm_configured = True
                    st.info(f"✅ Configured Gemini model: {gemini_model}")
            elif ollama_model:
                # Use optimized settings for document structuring
                if st.session_state.llm_handler.configure_ollama(
                    ollama_model,
                    temperature=0.3,  # Optimal for structured output
                    max_tokens=4000   # Sufficient for complex documents
                ):
                    llm_configured = True
                    st.info(f"✅ Configured Ollama model: {ollama_model} (optimized for JSON generation)")
        
        if not llm_configured:
            st.error("Could not configure the selected LLM. Please check API keys (for Gemini) or ensure Ollama is running. Processing halted.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            processed_docs = []
            total_files = len(uploaded_files)
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # MODIFIED CALL: Pass the configured llm_handler to the processor
                    docs = st.session_state.document_processor.process_file(
                        tmp_path, 
                        uploaded_file.name,
                        llm_handler=st.session_state.llm_handler
                    )
                    processed_docs.extend(docs)
                    
                    st.success(f"✅ {uploaded_file.name}: {len(docs)} structured items extracted")
                    os.unlink(tmp_path)
                    
                except Exception as e:
                    st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                    logger.error(f"Error processing {uploaded_file.name}: {e}", exc_info=True)
                
                progress_bar.progress((i + 1) / total_files)
            
            if processed_docs:
                with st.spinner("Building search index..."):
                    try:
                        st.session_state.search_engine = HybridSearchRAG(
                            documents=processed_docs,
                            vector_weight=vector_weight,
                            bm25_weight=bm25_weight,
                            top_k=top_k,
                            device=device
                        )
                        st.session_state.documents = processed_docs
                        
                        st.success(f"🎉 Successfully indexed {len(processed_docs)} document items!")
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Documents", len(uploaded_files))
                        col2.metric("Total Indexed Items", len(processed_docs))
                        if processed_docs:
                            avg_length = sum(len(doc['text']) for doc in processed_docs) // len(processed_docs)
                            col3.metric("Avg Item Length", f"{avg_length} chars")
                        
                    except Exception as e:
                        st.error(f"❌ Error building search index: {str(e)}")
                        logger.error(f"Error building search index: {e}", exc_info=True)
            
            status_text.text("✅ Processing complete!")

# Chat interface
st.header("💬 Chat Interface")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            with st.expander("📚 Sources", expanded=False):
                for i, source in enumerate(message["sources"], 1):
                    st.write(f"**Source {i}** (Score: {source['score']:.3f})")
                    st.code(f"{source['text'][:300]}{'...' if len(source['text']) > 300 else ''}", language='text')

if prompt := st.chat_input("Ask a question about your documents..."):
    if not st.session_state.search_engine:
        st.error("Please upload and process documents first!")
    else:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # The LLM is already configured from the document processing step,
                    # but we ensure it's set for the query processor.
                    model_choice = "gemini" if model_type == "ONLINE (Gemini)" else "ollama"
                    
                    result = st.session_state.query_processor.process_query(
                        query=prompt,
                        search_engine=st.session_state.search_engine,
                        llm_handler=st.session_state.llm_handler,
                        model_choice=model_choice,
                        min_relevance_score=min_relevance
                    )
                    
                    response = result.get("answer", "Sorry, I couldn't generate a response.")
                    st.write(response)
                    
                    sources = []
                    if "matched_documents" in result and "relevance_scores" in result:
                        sources = [{"text": doc, "score": score} 
                                   for doc, score in zip(result["matched_documents"], result["relevance_scores"])]

                    assistant_message = {
                        "role": "assistant",
                        "content": response,
                        "sources": sources,
                        "timestamp": time.time()
                    }
                    st.session_state.chat_history.append(assistant_message)
                    
                    if sources:
                        with st.expander("📚 Sources", expanded=False):
                            for i, source in enumerate(sources, 1):
                                st.write(f"**Source {i}** (Score: {source['score']:.3f})")
                                st.code(f"{source['text'][:300]}{'...' if len(source['text']) > 300 else ''}", language='text')
                
                except Exception as e:
                    error_msg = f"Error processing your question: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg, "sources": []})
                    logger.error(f"Error in chat processing: {e}", exc_info=True)

if st.session_state.chat_history:
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

st.markdown("---")
st.markdown("Developed with an intelligent document structuring pipeline.")