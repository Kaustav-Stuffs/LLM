from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
import os
import json
import csv
import google.generativeai as genai
import uvicorn
import logging
from webcamaccess import process_query
from RAG1 import HybridSearchRAG
from langchain_ollama import ChatOllama

# ----------------------------
# Logging Setup (Full DEBUG)
# ----------------------------
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ----------------------------
# Gemini Setup
# ----------------------------
genai.configure(api_key="AIzaSyDxYogQnjYmTbvghsqSWCKtot5DpygC8hA")
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# ----------------------------
# Qwen2.5 Setup
# ----------------------------
qwen_model = ChatOllama(model="phi3:3.8b", temperature=0.1, max_tokens=128)
logger.debug(f"Qwen2.5 model loaded: {qwen_model}")

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI()

# ----------------------------
# Request and Response Models
# ----------------------------
class QueryRequest(BaseModel):
    question: str
    userId: int
    model_choice: str = "llama"  # CHOOSE BETWEEN "gemini" OR "llama"

class QueryResponse(BaseModel):
    question: str
    userId: int
    answer: str
    model_used: str  # Include the model used in the response

# ----------------------------
# Read and Extract Context
# ----------------------------
def extract_context_from_files(file_paths: List[str]) -> List[dict]:
    documents = []
    for path in file_paths:
        logger.debug(f"Reading file: {path}")
        try:
            if path.endswith(".json"):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        documents.extend(data)
                    else:
                        documents.append(data)
            elif path.endswith(".txt"):
                with open(path, 'r', encoding='utf-8') as f:
                    documents.append({"text": f.read(), "metadata": {}})
            elif path.endswith(".csv"):
                with open(path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    text = "\n".join([", ".join(row) for row in reader])
                    documents.append({"text": text, "metadata": {}})
        except Exception as e:
            logger.error(f"Error reading file {path}: {str(e)}")
    return documents

# ----------------------------
# API Endpoint
# ----------------------------
@app.post("/customerSupport", response_model=QueryResponse)
async def customer_support(query: QueryRequest):
    import time
    start_time = time.time()
    logger.debug(f"Received request from user {query.userId} with question: {query.question}, model_choice: {query.model_choice}")

    # Load documents and initialize RAG for each request
    DOCUMENT_PATHS = ["./doc1.json"]
    documents = extract_context_from_files(DOCUMENT_PATHS)
    if not documents:
        logger.error("No documents loaded from doc1.json")
        return QueryResponse(
            question=query.question,
            userId=query.userId,
            answer="Sorry, something went wrong while loading documents.",
            model_used=query.model_choice
        )

    # Initialize HybridSearchRAG with the latest documents
    try:
        search_engine = HybridSearchRAG(
            documents=documents,
            vector_weight=0.7,
            bm25_weight=0.3,
            top_k=3
        )
    except Exception as e:
        logger.error(f"Error initializing HybridSearchRAG: {str(e)}")
        return QueryResponse(
            question=query.question,
            userId=query.userId,
            answer="Sorry, something went wrong while initializing search.",
            model_used=query.model_choice
        )

    # Process query using webcamaccess.py
    result = process_query(
        query.question,
        min_relevance_score=0.5,
        model_choice=query.model_choice,
        gemini_model=gemini_model,
        qwen_model=qwen_model,
        search_engine=search_engine
    )
    response_time = time.time() - start_time
    logger.debug(f"Response time: {response_time:.2f} seconds")
    return QueryResponse(
        question=query.question,
        userId=query.userId,
        answer=result["answer"],
        model_used=query.model_choice
    )

# ----------------------------
# App Entry Point
# ----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5556)