import os
import json
import csv
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from docx import Document
from pathlib import Path
import uuid
from rag_search import HybridSearchRAG

# Configuration
DATA_DIR = "/home/kaustav/AIML/NLP/data"
MODEL_NAME = "qwen2.5:1.5b"
# MODEL_NAME= "llama3.2:3b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
PORT = 5556
CHAT_HISTORY_LIMIT = 5
ENABLE_CHAT_HISTORY = True
ENABLE_CHUNK_SEARCH = True
MIN_RELEVANCE_SCORE = 0.5

# Global variables
hybrid_search = None
documents = []
all_chunks = []
all_chunk_metadata = []

def load_documents():
    global documents
    documents = []
    Path(DATA_DIR).mkdir(exist_ok=True)
    
    for file_path in Path(DATA_DIR).glob("*"):
        if file_path.suffix.lower() in [".txt", ".docx", ".csv", ".json"]:
            try:
                if file_path.suffix.lower() == ".txt":
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                elif file_path.suffix.lower() == ".docx":
                    doc = Document(file_path)
                    content = "\n".join([para.text for para in doc.paragraphs])
                elif file_path.suffix.lower() == ".csv":
                    with open(file_path, "r", encoding="utf-8") as f:
                        reader = csv.reader(f)
                        content = "\n".join([",".join(row) for row in reader])
                elif file_path.suffix.lower() == ".json":
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        content = json.dumps(data, indent=2)
                
                # Generate unique ID for each document
                doc_id = str(uuid.uuid4())
                documents.append({
                    "id": doc_id,
                    "text": content,
                    "metadata": {"path": str(file_path)}
                })
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")

def initialize_search():
    global hybrid_search, documents, all_chunks, all_chunk_metadata
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1800, chunk_overlap=300)
    all_chunks = []
    all_chunk_metadata = []
    chunked_documents = []

    if ENABLE_CHUNK_SEARCH:
        for doc in documents:
            splits = text_splitter.split_text(doc["text"])
            for idx, chunk in enumerate(splits):
                chunk_id = f"{doc['id']}_{idx}"
                chunked_documents.append({
                    "id": chunk_id,
                    "text": chunk,
                    "metadata": {
                        "path": doc["metadata"]["path"],
                        "chunk_index": idx,
                        "original_doc_id": doc["id"]
                    }
                })
                all_chunks.append(chunk)
                all_chunk_metadata.append({
                    "path": doc["metadata"]["path"],
                    "chunk_index": idx
                })
    else:
        for doc in documents:
            chunked_documents.append({
                "id": doc["id"],
                "text": doc["text"],
                "metadata": {
                    "path": doc["metadata"]["path"],
                    "chunk_index": 0,
                    "original_doc_id": doc["id"]
                }
            })
            all_chunks.append(doc["text"])
            all_chunk_metadata.append({
                "path": doc["metadata"]["path"],
                "chunk_index": 0
            })

    hybrid_search = HybridSearchRAG(
        documents=chunked_documents,
        embedding_model_name=EMBEDDING_MODEL,
        vector_weight=0.6,
        bm25_weight=0.4,
        top_k=3
    )

def perform_hybrid_search(query: str, k: int = 3) -> List[Dict]:
    print("[DEBUG] Starting hybrid search...")
    results = hybrid_search.search(query, min_relevance_score=MIN_RELEVANCE_SCORE)
    formatted_results = []
    for result in results:
        formatted_results.append({
            "content": result["text"],
            "score": result["relevance_score"],
            "metadata": result["metadata"]
        })
    # print(f"[DEBUG] Hybrid search results: {formatted_results}")
    return formatted_results[:k]