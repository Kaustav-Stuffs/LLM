from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
import os
import json
import csv
import fitz  
import google.generativeai as genai
import uvicorn
import logging
from webcamaccess import *  
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
qwen_model = ChatOllama(model="qwen2.5:1.5b", temperature=0.1, max_tokens=128)

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
    model_choice: str = "qwen"  # CHOOSE BETWEEN "gemini" OR "qwen"

class QueryResponse(BaseModel):
    question: str
    userId: int
    answer: str
    model_used: str  # Include the model used in the response

# ----------------------------
# Read and Extract Context
# ----------------------------
def extract_context_from_files(file_paths: List[str]) -> str:
    context_parts = []
    for path in file_paths:
        logger.debug(f"Reading file: {path}")
        try:
            if path.endswith(".txt"):
                with open(path, 'r', encoding='utf-8') as f:
                    context_parts.append(f.read())

            elif path.endswith(".json"):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    context_parts.append(json.dumps(data))

            elif path.endswith(".csv"):
                with open(path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    context_parts.append("\n".join([", ".join(row) for row in reader]))

            elif path.endswith(".pdf"):
                doc = fitz.open(path)
                text = ""
                for page in doc:
                    text += page.get_text()
                context_parts.append(text)
        except Exception as e:
            logger.error(f"Error reading file {path}: {str(e)}")

    combined_context = "\n".join(context_parts)
    # logger.debug(f"Combined context from documents:\n{combined_context}")
    return combined_context

# ----------------------------
# Load Document(s)
# ----------------------------
DOCUMENT_PATHS = ["/home/kaustav/AIML/NLP/data/doc1.txt"]
CONTEXT = extract_context_from_files(DOCUMENT_PATHS)

# ----------------------------
# API Endpoint
# ----------------------------
@app.post("/customerSupport", response_model=QueryResponse)
async def customer_support(query: QueryRequest):
    logger.debug(f"Received request from user {query.userId} with question: {query.question}, model_choice: {query.model_choice}")

    # Process query using webcamaccess.py
    result = process_query(
        query.question,
        min_relevance_score=0.5,
        model_choice=query.model_choice,
        gemini_model=gemini_model,
        qwen_model=qwen_model
    )
    
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