# General-Purpose RAG Chatbot

A sophisticated RAG (Retrieval Augmented Generation) chatbot built with Streamlit that offers advanced document processing and intelligent context-aware interactions across multiple file formats.

## Features

- **Multi-format Document Support**: PDF, DOCX, TXT, JSON, CSV
- **Hybrid Search**: Combines vector embeddings with BM25 lexical search
- **Dual LLM Support**: Both online (Gemini) and offline (Ollama) models
- **Enhanced Greeting Detection**: Smart handling of greetings with typo tolerance
- **Device Optimization**: Automatic detection of CUDA/MPS/CPU capabilities
- **Interactive Web Interface**: Built with Streamlit

## Installation

1. **Install dependencies:**
```bash
pip install google-genai langchain-ollama pypdf2 python-docx rank-bm25 scikit-learn streamlit torch
```

2. **Set up environment variables:**
```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
```

3. **For Ollama support (optional):**
   - Install Ollama: https://ollama.ai/
   - Pull models: `ollama pull llama2` or `ollama pull mistral`

## Usage

1. **Run the application:**
```bash
streamlit run app.py --server.port 5000
```

2. **Upload documents** using the file uploader in the sidebar

3. **Configure settings:**
   - Choose between Gemini (online) or Ollama (offline) models
   - Adjust search weights and relevance thresholds
   - Monitor device usage (CPU/GPU/MPS)

4. **Start chatting** with your documents!

## Project Structure

```
rag-chatbot/
├── app.py                          # Main Streamlit application
├── utils/
│   ├── __init__.py
│   ├── device_detector.py          # Hardware detection
│   ├── document_processor.py       # Multi-format document processing
│   ├── hybrid_search.py           # Vector + BM25 search engine
│   ├── llm_handler.py             # Gemini & Ollama integration
│   └── query_processor.py         # Enhanced query processing
├── .streamlit/
│   └── config.toml                # Streamlit configuration
├── pyproject.toml                 # Project dependencies
└── README.md                      # This file
```

## Key Components

### Enhanced Greeting Detection
- Handles typos like "hib" → "hi", "helo" → "hello"
- Distinguishes between greeting-only vs greeting+content queries
- Natural, varied responses for greetings and closings

### Hybrid Search Engine
- **Vector Search**: TF-IDF embeddings with cosine similarity
- **Lexical Search**: BM25 for exact keyword matching
- **Combined Scoring**: Configurable weight balancing
- **Relevance Filtering**: Minimum score thresholds

### Multi-format Processing
- **PDF**: PyPDF2 or pdfplumber support
- **DOCX**: Table and paragraph extraction
- **TXT**: UTF-8 and Latin-1 encoding support
- **JSON**: Structured data with automatic ID detection
- **CSV**: Header-aware row processing

### Device Optimization
- **CUDA**: GPU acceleration for NVIDIA cards
- **MPS**: Apple Silicon optimization
- **CPU**: Fallback with threading support

## Configuration

### Model Selection
- **Gemini**: Requires `GEMINI_API_KEY` environment variable
- **Ollama**: Requires local Ollama installation

### Search Parameters
- **Vector Weight**: Semantic similarity importance (0.0-1.0)
- **BM25 Weight**: Keyword matching importance (auto-calculated)
- **Top K**: Number of search results to consider
- **Min Relevance**: Threshold for result filtering

## Architecture Highlights

1. **Document-Only Responses**: Answers based strictly on uploaded content
2. **Session Persistence**: Chat history and document storage
3. **Error Handling**: Comprehensive error management and logging
4. **Modular Design**: Clean separation of concerns
5. **Performance Optimization**: Efficient indexing and search

## Requirements

- Python 3.11+
- Streamlit 1.47.1+
- PyTorch 2.7.1+
- scikit-learn 1.7.1+
- Optional: CUDA-capable GPU for acceleration

## License

Open source - feel free to modify and distribute.

## Support

For issues or questions, please check the logs in the Streamlit interface or review the error messages in the chat interface.