import os
import json
import csv
import time
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from docx import Document
from pathlib import Path
import uvicorn
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
import torch
import uuid




# Configuration
DATA_DIR = "/home/kaustav/AIML/NLP/data"
MODEL_NAME = "qwen2.5:1.5b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
PORT = 5556
CHAT_HISTORY_LIMIT = 5

# Switches to turn ON/OFF chat history and chunk search
ENABLE_CHAT_HISTORY = True
ENABLE_CHUNK_SEARCH = True
MIN_RELEVANCE_SCORE = 0.5
# HybridSearchRAG class from RAG1.py
class HybridSearchRAG:
    """
    Hybrid Search implementation for Retrieval Augmented Generation.
    Combines vector-based semantic search with BM25 lexical search.
    """
    def __init__(
            self,
            documents: List[Dict],
            embedding_model_name: str = "all-MiniLM-L6-v2",
            vector_weight: float = 0.7,
            bm25_weight: float = 0.3,
            top_k: int = 5
    ):
        self.documents = documents
        self.texts = [doc['text'] for doc in documents]
        self.doc_ids = [doc['id'] for doc in documents]
        self.top_k = top_k

        # Ensure weights sum to 1
        total = vector_weight + bm25_weight
        self.vector_weight = vector_weight / total
        self.bm25_weight = bm25_weight / total

        # Initialize vector search
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.document_embeddings = self._create_document_embeddings()

        # Initialize BM25 search
        print("Building BM25 index...")
        self.bm25_tokenized_corpus = [text.lower().split() for text in self.texts]
        self.bm25 = BM25Okapi(self.bm25_tokenized_corpus)

    def _create_document_embeddings(self) -> np.ndarray:
        print("Creating document embeddings...")
        return self.embedding_model.encode(self.texts, show_progress_bar=True)

    def _vector_search(self, query: str, top_k: int = None) -> List[tuple[int, float]]:
        if top_k is None:
            top_k = self.top_k
        query_embedding = self.embedding_model.encode(query)
        similarities = cosine_similarity([query_embedding], self.document_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(idx, similarities[idx]) for idx in top_indices]

    def _bm25_search(self, query: str, top_k: int = None) -> List[tuple[int, float]]:
        if top_k is None:
            top_k = self.top_k
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        return [(idx, bm25_scores[idx]) for idx in top_indices]

    def search(self, query: str, reranker: Optional[callable] = None, min_relevance_score: float = 0.5) -> List[Dict]:
        query_keywords = set(query.lower().split()) - set(['is', 'are', 'there', 'any', 'in', 'this', 'the', 'a', 'an'])
        has_keyword_match = False
        for text in self.texts:
            text_lower = text.lower()
            if any(keyword in text_lower for keyword in query_keywords):
                has_keyword_match = True
                break
        if not has_keyword_match:
            return []

        vector_results = self._vector_search(query, top_k=self.top_k * 2)
        bm25_results = self._bm25_search(query, top_k=self.top_k * 2)

        def normalize_scores(results):
            scores = [score for _, score in results]
            if not scores or max(scores) == min(scores):
                return [(idx, 0.0) for idx, _ in results]
            score_range = max(scores) - min(scores)
            return [(idx, (score - min(scores)) / score_range if score_range > 0 else 0.0)
                    for idx, score in results]

        vector_results = normalize_scores(vector_results)
        bm25_results = normalize_scores(bm25_results)

        combined_scores = {}
        for idx, score in vector_results:
            combined_scores[idx] = self.vector_weight * score
        for idx, score in bm25_results:
            if idx in combined_scores:
                combined_scores[idx] += self.bm25_weight * score
            else:
                combined_scores[idx] = self.bm25_weight * score

        top_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:self.top_k]
        results = []
        for idx, score in top_results:
            if score >= min_relevance_score:
                doc = self.documents[idx].copy()
                doc['relevance_score'] = float(score)
                results.append(doc)

        if reranker and callable(reranker) and results:
            results = reranker(query, results)
        return results

# Lifespan handler for startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    load_documents()
    if documents:
        initialize_search()
    else:
        print("No documents found in data directory")
    yield
    print("Shutting down application")

# Initialize FastAPI app with lifespan
app = FastAPI(title="Document QA API", lifespan=lifespan)



# Initialize LLM
llm = ChatOllama(model=MODEL_NAME, temperature=0.18, max_tokens=128)

# Global variables
hybrid_search = None
documents = []
chat_history = []
all_chunks = []
all_chunk_metadata = []

# Pydantic model for request
class QuestionRequest(BaseModel):
    question: str
    userId: int

# Function to load documents
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

# Function to initialize search indices
def initialize_search():
    global hybrid_search, documents, all_chunks, all_chunk_metadata
    # Adjust chunk size and overlap as needed for your scenario
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
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
        # Use whole documents as single chunks
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

# Hybrid search function using HybridSearchRAG
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

# Prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are a professional customer support assistant. So be polite and helpful.
    You are not allowed to use any external knowledge or information that is not provided in the document context.
    You are given a set of documents to answer questions.
    Your task is to answer questions based strictly on the provided document context. 
    Don't use any external knowledge or information (on your own) that is not provided in the document context.
    Use the chat history to understand context but only answer based on the document content.
    If the question is out of document context or cannot be answered with the provided documents, strictly respond with:
    "I couldn't understand your question. Could you please rephrase it more clearly?"
    
    Chat History:
    {chat_history}
    
    Document Context:
    {context}
    
    Question: {question}"""),
    ("human", "{question}")
])

# API endpoint
@app.post("/customerSupport")
async def ask_question(request: QuestionRequest):
    global chat_history
    start_time = time.time()
    question = request.question
    user_id = request.userId

    print(f"[DEBUG] Received question: \n {question} \n from user: {user_id}")
    search_results = perform_hybrid_search(question)
    
    # Check if top result meets MIN_RELEVANCE_SCORE
    if not search_results or search_results[0].get("score", 0.0) < MIN_RELEVANCE_SCORE:
        print("\nTop results:")
        print("No relevant results found for the query or confidence is too low.")
        answer = "I couldn't understand your question. Could you please rephrase it more clearly?"
        response_time = round(time.time() - start_time, 3)
        return JSONResponse(content={
            "data": {
                "question": question,
                "userId": user_id,
                "answer": answer
            },
            "is_json": True,
            "message": "Response Code:200.",
            "response_time": response_time
        })

    # Build context from Top Results only (concatenate their text)
    context_chunks = []
    seen = set()
    for result in search_results:
        key = (result["metadata"]["path"], result["metadata"]["chunk_index"])
        if key not in seen:
            context_chunks.append(result['content'])
            seen.add(key)
    context = "\n\n".join(context_chunks)

    # Print top results in a beautified format for DEBUG
    print("\nTop results:")
    for i, result in enumerate(search_results):
        score = result.get("score", 1.0)
        content = result.get("content", "")
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict) and "text" in parsed:
                display_text = parsed["text"]
            else:
                display_text = json.dumps(parsed, indent=2)
        except Exception:
            display_text = content.strip().replace("\n", " ")
        print(f"{i + 1}. [Score: {score:.4f}] {display_text}\n{'-'*80}")

    # Format chat history
    history_str = ""
    if ENABLE_CHAT_HISTORY:
        print("[DEBUG] Formatting chat history...")
        for msg in chat_history[-CHAT_HISTORY_LIMIT:]:
            role = "Human" if isinstance(msg, HumanMessage) else "Assistant"
            history_str += f"{role}: {msg.content}\n"
    # else: history_str remains empty

    # Create prompt and get response
    # The prompt template already instructs to answer strictly from context.
    # To enforce, we only provide the Top Results as context.
    chain = prompt_template | llm
    print("[DEBUG] Created LLM chain. Invoking LLM...")
    try:
        response = await chain.ainvoke({
            "question": question,
            "context": context,
            "chat_history": history_str
        })
        answer = response.content
        print(f"[DEBUG] LLM response: {answer}")
    except Exception as e:
        print(f"[DEBUG] Exception during LLM invocation: {e}")
        answer = "I couldn't understand your question. Could you please rephrase it more clearly?"

    # Update chat history
    if ENABLE_CHAT_HISTORY:
        print(f"[DEBUG] Updating chat history with question and answer.")
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))
        if len(chat_history) > CHAT_HISTORY_LIMIT * 2:
            chat_history = chat_history[-CHAT_HISTORY_LIMIT * 2:]

    # Return response
    response_time = round(time.time() - start_time, 3)
    print(f"[DEBUG] Returning response. Response time: {response_time} seconds")
    return JSONResponse(content={
        "data": {
            "question": question,
            "userId": user_id,
            "answer": answer
        },
        "is_json": True,
        "message": "Response Code:200.",
        "response_time": response_time
    })

# Main function to run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)