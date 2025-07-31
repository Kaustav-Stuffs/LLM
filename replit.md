# RAG Chatbot Application

## Overview

This is a Retrieval Augmented Generation (RAG) chatbot application built with Streamlit that allows users to upload documents and interact with them through natural language queries. The system combines semantic search with lexical search to provide accurate and contextually relevant responses using multiple LLM providers.

## User Preferences

Preferred communication style: Simple, everyday language.
Enhanced greeting detection: User requested improved handling of mistyped greetings and closings with fuzzy matching.

## System Architecture

The application follows a modular architecture with clear separation of concerns:

- **Frontend**: Streamlit-based web interface for user interactions
- **Document Processing**: Multi-format document ingestion and chunking system
- **Search Engine**: Hybrid search combining BM25 lexical search with vector embeddings
- **LLM Integration**: Support for multiple language models (Gemini, Ollama)
- **State Management**: Session-based state management through Streamlit

## Key Components

### 1. Main Application (`app.py`)
- **Purpose**: Entry point and UI coordination
- **Key Features**: 
  - Session state management for chat history and documents
  - Thread-safe search engine initialization
  - File upload and processing interface
  - Chat interface with message history

### 2. Document Processor (`document_processor.py`)
- **Purpose**: Handle multiple document formats and convert them to searchable chunks
- **Supported Formats**: PDF (PyPDF2/pdfplumber), DOCX, TXT, JSON, CSV
- **Architecture Decision**: Flexible processor with optional dependencies to handle missing libraries gracefully
- **Chunking Strategy**: Configurable chunk size (1000 chars) with overlap (200 chars) for context preservation

### 3. RAG Engine (`rag_engine.py`)
- **Purpose**: Implement hybrid search combining semantic and lexical search
- **Architecture Decision**: Chose hybrid approach over pure vector search for better handling of exact matches and semantic queries
- **Components**:
  - BM25 for lexical search
  - SentenceTransformer for semantic embeddings
  - Weighted scoring system (70% vector, 30% BM25)
- **Device Support**: Automatic detection of CUDA, MPS, or CPU for optimal performance

### 4. LLM Handler (`llm_handler.py`)
- **Purpose**: Abstract interface for LLM providers with intelligent conversation handling
- **Supported Models**: Google Gemini (primary), fallback responses when unavailable
- **Architecture Decision**: Provider abstraction with integrated greeting/closing detection
- **Key Features**: 
  - Fuzzy matching for mistyped greetings and farewells
  - Context-aware response generation
  - Automatic greeting prefix addition for natural conversations

### 5. Greeting Handler (`greeting_handler.py`)
- **Purpose**: Advanced greeting and closing detection with typo tolerance
- **Key Features**:
  - Fuzzy string matching using SequenceMatcher
  - Supports various greeting formats (hi, hello, hey, good morning, etc.)
  - Handles common misspellings and variations
  - Distinguishes between greeting-only and greeting+question inputs
  - Natural conversation flow management

### 6. Utilities (`utils.py`)
- **Purpose**: Helper functions for device detection, text processing, and formatting
- **Key Functions**: Device info detection, file size formatting, text truncation
- **Architecture Decision**: Simplified for offline operation without external dependencies

## Data Flow

1. **Document Upload**: Users upload files through Streamlit interface
2. **Processing**: DocumentProcessor converts files to text chunks with metadata
3. **Indexing**: HybridSearchRAG creates BM25 index and vector embeddings
4. **Query Processing**: User queries are processed through hybrid search
5. **Context Retrieval**: Top-k relevant chunks are retrieved and ranked
6. **Response Generation**: LLM generates response using retrieved context
7. **Display**: Response is shown in chat interface with source attribution

## External Dependencies

### Required Dependencies
- **streamlit**: Web interface framework
- **google-genai**: Gemini API integration
- **numpy**: Numerical computations (available by default)

### Current Architecture (Offline-First)
- **Simple RAG Engine**: Custom lexical search without external ML libraries
- **Built-in Python Libraries**: json, csv, re, collections for core functionality
- **Fallback Mechanisms**: Graceful degradation when advanced features unavailable

### Previously Attempted (Dependency Conflicts)
- **sentence-transformers**: Semantic embeddings (installation conflicts)
- **rank-bm25**: Lexical search implementation (dependency issues)
- **scikit-learn**: Similarity calculations (conflicts with PyTorch configuration)

### Architecture Decision Rationale
Switched to offline-first approach due to complex dependency conflicts in the Replit environment. The current implementation provides core RAG functionality using only essential dependencies while maintaining extensibility for future enhancements.

## Deployment Strategy

### Development Environment
- Local development with Streamlit development server
- Environment variables for API keys (GEMINI_API_KEY)
- Hot reloading for rapid iteration

### Production Considerations
- **Caching**: Uses Streamlit's @st.cache_resource for expensive operations
- **Thread Safety**: Global locks for search engine operations
- **Memory Management**: Efficient document chunking and embedding storage
- **Error Handling**: Graceful degradation with comprehensive logging

### Scalability Considerations
- **Search Engine**: In-memory lexical search suitable for moderate document volumes
- **Future Enhancement**: Can be extended with vector databases when dependency issues resolved
- **Multi-user Support**: Current architecture supports single-user sessions

### Current Deployment Status
- **Environment**: Streamlit on Replit (Python 3.11)
- **Port**: 5000 (configured for Replit deployment)
- **Dependencies**: Minimal set to avoid conflicts
- **File Support**: TXT, JSON, CSV (PDF/DOCX disabled due to dependency issues)

## Recent Changes

### January 28, 2025
- Enhanced greeting detection with fuzzy matching for typos and variations
- Integrated GreetingHandler into LLM pipeline for natural conversation flow
- Added comprehensive greeting/closing pattern recognition
- Improved user experience for conversational interactions
- Created sample document for testing functionality