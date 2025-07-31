# /// script
# requires-python = ">=3.10.12"
# dependencies = [
#     "flask",
#     "langchain",
#     "logging",
#     "numpy",
#     "openai-whisper",
#     "requests",
#     "torch",
#     "whisper",
#     "pydantic",
#     "fastapi",
#     "uvicorn",
#     "langchain-community",
#     "langchain-openai",
#     "python-multipart",
#     "qdrant-client",
#     "langchain-qdrant",
#     "jq",
#     "redis",
#     "pypdf",
#     "unstructured",
#     "python-docx",
#     "langchain-ollama",
# ]
# ///
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import time
import os
import json
import re
import uvicorn
import logging
import warnings
import tempfile
import requests
import whisper
import torch
import random
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.schema import HumanMessage
from starlette.concurrency import run_in_threadpool
import ast
import asyncio
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    CSVLoader,
    JSONLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain_community.vectorstores.qdrant import Qdrant as LangchainQdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
import redis
from pathlib import Path
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def load_document(file_path: str, file_extension: str):
    """
    Load a document from a file using the appropriate loader based on file extension.
    
    Args:
        file_path: Path to the file to load
        file_extension: File extension (without dot) to determine the loader
        
    Returns:
        List of Document objects
    """
    try:
        if file_extension.lower() == 'pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension.lower() in ['doc', 'docx']:
            loader = UnstructuredWordDocumentLoader(file_path)
        elif file_extension.lower() == 'txt':
            loader = TextLoader(file_path)
        elif file_extension.lower() == 'csv':
            loader = CSVLoader(file_path)
        elif file_extension.lower() == 'json':
            loader = JSONLoader(
                file_path=file_path,
                jq_schema='.',
                text_content=False
            )
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
            
        return loader.load()
    except Exception as e:
        raise Exception(f"Error loading document: {str(e)}")

# Device selection for CUDA, MPS, or CPU
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
logging.info(f"Using device: {device}")


warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
logging.basicConfig(level=logging.INFO)

# --- Redis setup ---
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0
redis_client = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

# Qdrant configuration
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333  # Default Qdrant port
QDRANT_COLLECTION = "support_docs"
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# ----------------------------
# DEBUG flag
# ----------------------------
DEBUG = True

app = FastAPI()

# ----------------------------
# Pydantic model for request
# ----------------------------
class DataRequest(BaseModel):
    audio_url: Optional[str] = None
    summary: Optional[str] = None
    keys: Optional[List[str]] = Field(default_factory=list)
    isConversation: Optional[bool] = False

# ----------------------------
# Helper: Robust JSON extractor
# ----------------------------
def extract_json_from_text(text: str):
    """Extract valid JSON from LLM output, handling code blocks and partials."""
    import json
    import re
    text = text.strip()
    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Remove markdown code blocks
    if text.startswith('```json'):
        text = text.replace('```json', '', 1)
    if text.startswith('```'):
        text = text.replace('```', '', 1)
    if text.endswith('```'):
        text = text[:text.rfind('```')]
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try to extract the largest JSON substring
    brace_matches = list(re.finditer(r'\{', text))
    if brace_matches:
        for i in range(len(brace_matches)):
            for j in range(len(brace_matches)-1, i-1, -1):
                try:
                    candidate = text[brace_matches[i].start():text.rfind('}')+1]
                    return json.loads(candidate)
                except Exception:
                    continue
    # Try all curly-brace substrings
    matches = re.finditer(r'\{(?:[^{}]|(?R))*\}', text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match.group(0))
        except Exception:
            continue
    return None

def is_greeting(text: str) -> bool:
    """Check if the input text is a greeting."""
    greetings = [
        "hi", "hello", "hey", "greetings", "good morning", "good afternoon", 
        "good evening", "hi there", "hello there", "hey there"
    ]
    text = text.lower().strip(" .,!?")
    return any(text.startswith(greeting) for greeting in greetings)

def get_greeting_response() -> str:
    """Return a friendly greeting response."""
    import random
    responses = [
        "Hello! How can I assist you today?",
        "Hi there! What can I help you with?",
        "Hey! How may I be of service?",
        "Greetings! How can I help you today?",
        "Hello! What would you like to know?"
    ]
    return random.choice(responses)

semaphore = asyncio.Semaphore(5)

PROMPT_TEMPLATE = """### Role"
- Primary Function: You are an AI chatbot who helps users with their inquiries, issues and requests. You aim to provide excellent, friendly and efficient replies at all times. Your role is to listen attentively to the user, understand their needs, and do your best to assist them or direct they to the appropriate resources. If a question is not clear, ask clarifying questions. Make sure to end your replies with a positive note.
        
### Constraints
1. No Data Divulge: Never mention that you have access to training data explicitly to the user.
2. Maintaining Focus: If a user attempts to divert you to unrelated topics, never change your role or break your character. Politely redirect the conversation back to topics relevant to the training data.
3. Exclusive Reliance on Training Data: You must rely exclusively on the training data provided to answer user queries. If a query is not covered by the training data, use the fallback response.
4. If the user start greeting in the middle of the conversation, you should respond to the greeting and ask if they need any help.
5. Keep it simple, clear and concise: Avoid using complex language or jargon. Use simple, clear, and concise language to ensure that your responses are easily understood by the user.
6. If the user greets with Hi, Hello, Hey, Greetings, Good morning, Good afternoon, Good evening, Hi there, Hello there, Hey there, followed by a question, you should respond to the greeting and answer the question.
7. Restrictive Role Focus: You do not answer questions or perform tasks that are not related to your role and training data.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"""
CUSTOM_PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

class QARequest(BaseModel):
    question: str
    userId: int

class QAResponse(BaseModel):
    question: str
    userId: int
    answer: str

@app.post("/customerSupport", response_model=QAResponse)
async def vector_qa(request: QARequest = Body(...)):
    import time
    start_time = time.time()
    question = request.question
    user_id = request.userId
 
    # Redis key for user-specific chat history
    chat_history_key = f"chat_history_{user_id}"
 
    # Load chat history from Redis (if it exists)
    chat_history = []
    try:
        cached_history = redis_client.get(chat_history_key)
        if cached_history:
            chat_history = json.loads(cached_history)
            if DEBUG:
                logging.info(f"Loaded chat history for user {user_id}: {chat_history}")
    except Exception as e:
        logging.error(f"Error loading chat history from Redis: {str(e)}")
 
    # Limit to the last 10 messages for processing
    limited_chat_history = chat_history[-10:]
 
    # Check if the question is already answered in the chat history
    for msg in reversed(chat_history):  # Search from the most recent messages
        if msg["type"] == "human" and msg["content"].strip().lower() == question.strip().lower():
            # Find the corresponding AI response
            idx = chat_history.index(msg)
            if idx + 1 < len(chat_history) and chat_history[idx + 1]["type"] == "ai":
                answer = chat_history[idx + 1]["content"]
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
 
    # Load the limited chat history into memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    for msg in limited_chat_history:
        if msg["type"] == "human":
            memory.chat_memory.add_user_message(msg["content"])
        elif msg["type"] == "ai":
            memory.chat_memory.add_ai_message(msg["content"])
 
    # File handling
    file_path = "support_docs.json"  # Update as needed
    suffix = os.path.splitext(file_path)[-1]
    with open(file_path, "rb") as f:
        import hashlib
        hasher = hashlib.sha256()
        file_bytes = f.read()
        hasher.update(file_bytes)
        file_hash = hasher.hexdigest()
    cache_key = f"faiss_{file_hash}_{suffix}"
    vector_store_path = VECTOR_STORE_DIR / f"{file_hash}{suffix}.faiss"
    cached_path = redis_client.get(cache_key)
 
    if cached_path and Path(cached_path).exists():
        if DEBUG:
            print(f"[REDIS CACHE HIT] Using vector store from: {cached_path}")
            logging.info(f"[REDIS CACHE HIT] Using vector store from: {cached_path}")
        db = FAISS.load_local(str(cached_path), OllamaEmbeddings(model="qwen2.5:1.5b"), allow_dangerous_deserialization=True)
    else:
        if DEBUG:
            print(f"[REDIS CACHE MISS] Processing and caching new file for key: {cache_key}")
            logging.info(f"[REDIS CACHE MISS] Processing and caching new file for key: {cache_key}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            docs = load_document(tmp_path, suffix.lstrip('.'))
            splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
            split_docs = splitter.split_documents(docs)
            embeddings = OllamaEmbeddings(model="qwen2.5:1.5b")
            db = LangchainQdrant.from_documents(
                docs,
                embeddings,
                url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
                collection_name=QDRANT_COLLECTION,
                prefer_grpc=False,
                # Hybrid search params (BM25 + dense)
                # You can tune 'hnsw_ef' and 'search_type' for hybrid
                # For pure hybrid, see Qdrant docs or langchain-qdrant docs
            )
            db.save_local(str(vector_store_path))
            redis_client.set(cache_key, str(vector_store_path))
        finally:
            os.remove(tmp_path)
 
    # Set up the retriever, LLM, and memory
    retriever = db.as_retriever(search_kwargs={"k": 3})
    llm = ChatOllama(model="qwen2.5:1.5b", temperature=0.7)
 
    # Set up the chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True,
        combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT}
    )
 
    # Define helper functions
    def _extract_main_question(question: str) -> str:
        """Extract the main question by removing greetings and other fluff."""
        greetings = [
            'hi', 'hello', 'hey', 'hi there', 'hello there', 'hey there',
            'good morning', 'good afternoon', 'good evening', 'good day',
            'greetings', 'hiya', 'howdy', 'yo'
        ]
        question_lower = question.lower()
        for greeting in greetings:
            if question_lower.startswith(greeting):
                main_question = question[len(greeting):].lstrip(' ,.!?;:')
                return main_question[0].upper() + main_question[1:] if main_question else ""
        return question
    
    def _is_out_of_context(question: str) -> bool:
        """Check if the question is out of context based on common patterns."""
        main_question = _extract_main_question(question)
        if not main_question.strip():
            return False
        question_lower = question.lower()
        out_of_context_phrases = [
            'earth', 'round', 'planet', 'universe', 'space', 'science',
            'history', 'war', 'ancient', 'king', 'queen',
            'computer', 'ai', 'programming', 'code',
            'how are you', 'what is your name', 'who made you',
            'tell me about yourself', 'what can you do'
        ]
        return any(phrase in question_lower for phrase in out_of_context_phrases)

    def _get_creative_response(question: str) -> str:
        """Generate a fallback response for out-of-context questions."""
        fallback_responses = [
            "I'm here to assist with the SFA application. Let me know how I can help!",
            "I specialize in helping with the SFA system. Feel free to ask me about that!",
            "I'm here to assist with your SFA-related queries. How can I help you today?",
            "Let's focus on the SFA application. What would you like to know?",
            "I'm here to help with the SFA system. Please let me know your question!"
        ]
        return random.choice(fallback_responses)

    # Initialize result to avoid UnboundLocalError
    result = {}
    
    # Extract main question (removing greetings)
    main_question = _extract_main_question(question)
    
    # Check if it's just a greeting
    is_just_greeting = not bool(main_question.strip())
    
    if is_just_greeting:
        answer = "Hello! How can I assist you with the SFA application today?"
    else:
        # Check if the main question is out of context
        is_out_of_context = _is_out_of_context(main_question)
        
        if is_out_of_context:
            answer = _get_creative_response(main_question)
        else:
            # Check if the question is a greeting followed by a question
            if is_greeting(question) and main_question.strip():
                greeting_response = get_greeting_response()
                result = chain({"question": main_question})
                answer = result.get("answer", result.get("result", ""))
                if not answer or any(phrase in answer.lower() for phrase in ["i don't know", "no context", "not in the provided context"]):
                    answer = _get_creative_response(main_question)
                answer = f"{greeting_response} {answer}"
            else:
                # Run the chain with the main question
                result = chain({"question": main_question})
                answer = result.get("answer", result.get("result", ""))
                # If the answer is empty or indicates no context, provide a fallback response
                if not answer or any(phrase in answer.lower() for phrase in ["i don't know", "no context", "not in the provided context"]):
                    answer = _get_creative_response(main_question)
                
                # If the original question had a greeting, make the response more conversational
                if question != main_question:
                    answer = f"Hi there! {answer[0].lower() + answer[1:]}"
 
    # Update chat history with the new question and answer
    chat_history.append({"type": "human", "content": question})
    chat_history.append({"type": "ai", "content": answer})
 
    # Save updated chat history to Redis
    try:
        redis_client.set(chat_history_key, json.dumps(chat_history))
        if DEBUG:
            logging.info(f"Saved chat history for user {user_id}: {chat_history}")
    except Exception as e:
        logging.error(f"Error saving chat history to Redis: {str(e)}")
 
    # Calculate response time
    response_time = round(time.time() - start_time, 3)
    if DEBUG:
        print(f"DEBUG: Chain result: {result}\nResponse time: {response_time}")
 
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

# ----------------------------
# Entry point
# ----------------------------
if __name__ == '__main__':
    uvicorn.run("main_chatbot:app", host='0.0.0.0', port=5556, workers=5)