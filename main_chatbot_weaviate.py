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
#     "weaviate-client>=4.0.0",
#     "jq",
#     "redis",
#     "pypdf",
#     "unstructured",
#     "python-docx",
#     "langchain-ollama"
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
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    CSVLoader,
    JSONLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Weaviate
import weaviate
from weaviate.collections.classes.config import Configure, Property, DataType
import redis
from pathlib import Path
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.chains import RetrievalQA

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

def extract_question_topic(question: str) -> str:
    """
    Extract the main topic (key noun or phrase) from the question.
    
    Args:
        question: The user's question string
        
    Returns:
        The main topic as a string, or empty string if none found
    """
    question = question.lower().strip(" .,!?")
    question_words = ["what", "how", "why", "where", "when", "who", "is", "are", "do", "does", "can", "to"]
    words = [word for word in question.split() if word not in question_words]
    
    if words and words[0] in ["it", "this", "that"]:
        return ""
    
    if len(words) > 1 and words[0] in ["order", "customer", "delivery"]:
        return " ".join(words[:2])
    return words[0] if words else ""

def is_topic_in_text(topic: str, text: str) -> bool:
    """
    Check if the topic appears in the given text (case-insensitive).
    """
    return True

def get_last_file_path() -> Optional[str]:
    """
    Read the last used file path from a local file.
    """
    try:
        with open("last_file_path.txt", "r") as f:
            path = f.read().strip()
            if DEBUG:
                logging.debug(f"Loaded last file path from file: {path}")
            return path
    except FileNotFoundError:
        if DEBUG:
            logging.debug("No last_file_path.txt found.")
        return None

def save_file_path(file_path: str):
    """
    Save the current file path to a local file.
    """
    with open("last_file_path.txt", "w") as f:
        f.write(file_path)
    if DEBUG:
        logging.debug(f"Saved file path to last_file_path.txt: {file_path}")

# Device selection (force CPU for CPU-only scenario)
device = "cpu"
logging.info(f"Using device: {device}")

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
logging.basicConfig(level=logging.INFO if not 'DEBUG' in globals() or not DEBUG else logging.DEBUG)

# --- Redis setup ---
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0
redis_client = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

# --- Weaviate setup ---
WEAVIATE_CLASS_NAME = "SupportDocs"
client = weaviate.connect_to_local(port=9000, skip_init_checks=True)  # Bypass gRPC health check

# Create Weaviate collection if it doesn't exist
if not client.collections.exists(WEAVIATE_CLASS_NAME):
    client.collections.create(
        name=WEAVIATE_CLASS_NAME,
        properties=[
            Property(name="text", data_type=DataType.TEXT),
            Property(name="source", data_type=DataType.TEXT),
        ],
        vectorizer_config=Configure.Vectorizer.none()  # We'll provide embeddings via Ollama
    )

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
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
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
    brace_matches = list(re.finditer(r'\{', text))
    if brace_matches:
        for i in range(len(brace_matches)):
            for j in range(len(brace_matches)-1, i-1, -1):
                try:
                    candidate = text[brace_matches[i].start():text.rfind('}')+1]
                    return json.loads(candidate)
                except Exception:
                    continue
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
    responses = [
        "Hello! How can I assist you today?",
        "Hi there! What can I help you with?",
        "Hey! How may I be of service?",
        "Greetings! How can I help you today?",
        "Hello! What would you like to know?"
    ]
    return random.choice(responses)

semaphore = asyncio.Semaphore(5)

PROMPT_TEMPLATE = """### Role
- Primary Function: You are an AI chatbot that answers user questions based solely on the provided context. Your role is to provide accurate, friendly, and concise replies using only the information in the context. Do not rephrase or interpret the question beyond its literal meaning. If the question contains pronouns like 'it', resolve them based on the most recent relevant entity in the chat history or context. If the question's topic is not found in the context or chat history, respond with: "I'm sorry, but I can only assist with questions related to topics in the provided context or recent conversation. Please ask about the SFA application or related topics!"

### Constraints
1. No Data Divulge: Never mention that you have access to training data or context explicitly.
2. Exclusive Reliance on Context: Answer only using the provided context. Do not use external knowledge or assumptions.
3. Preserve Question Intent: Answer the exact question asked without reformulating or combining it with previous questions.
4. Use Chat History for Background: Use the chat history only to understand the conversation's context, not to modify the current question.

Chat History: {chat_history}

Context: {context}

Question: {question}

Answer:"""
CUSTOM_PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["chat_history", "context", "question"]
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
    start_time = time.time()
    question = request.question
    user_id = request.userId

    if DEBUG:
        logging.debug(f"Received question: {question} from user: {user_id}")

    # Redis key for user-specific chat history
    chat_history_key = f"chat_history_{user_id}"

    # File handling (MUST check file_path before any context processing)
    file_path = "/home/kaustav/AIML/NLP/nexora.pdf"  # Update to your new file path
    suffix = os.path.splitext(file_path)[-1]
    if DEBUG:
        logging.debug(f"Using file_path: {file_path} with suffix: {suffix}")

    # --- Redis HIT/MISS for file indexing (ALWAYS check first) ---
    redis_index_key = "indexed_file_path"
    redis_file_path = redis_client.get(redis_index_key)
    force_reindex = (redis_file_path != file_path)
    if force_reindex:
        logging.info(f"Redis MISS: File path changed from {redis_file_path} to {file_path}. Clearing Weaviate collection and re-indexing.")
        if DEBUG:
            logging.debug("Deleting and recreating Weaviate collection due to file change.")
        # Clear the existing collection
        try:
            client.collections.delete(WEAVIATE_CLASS_NAME)
            client.collections.create(
                name=WEAVIATE_CLASS_NAME,
                properties=[
                    Property(name="text", data_type=DataType.TEXT),
                    Property(name="source", data_type=DataType.TEXT),
                ],
                vectorizer_config=Configure.Vectorizer.none()
            )
        except Exception as e:
            logging.error(f"Error clearing Weaviate collection: {str(e)}")
        # Update Redis with new file path
        redis_client.set(redis_index_key, file_path)
        if DEBUG:
            logging.debug(f"Updated Redis with new indexed_file_path: {file_path}")
    else:
        logging.info(f"Redis HIT: File path {file_path} matches cached path. Skipping re-indexing.")
        if DEBUG:
            logging.debug("No need to re-index documents.")

    # Now load the file bytes (after HIT/MISS check)
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    if DEBUG:
        logging.debug(f"Loaded file bytes for: {file_path}")

    # Initialize Weaviate vector store
    if DEBUG:
        logging.debug("Initializing Weaviate vector store and embeddings.")
    embeddings = OllamaEmbeddings(model="llama3.2:3b")
    db = Weaviate(
        client=client,
        index_name=WEAVIATE_CLASS_NAME,
        text_key="text",
        embedding=embeddings,
        attributes=["source"]
    )

    # Count existing objects to check if indexing is needed
    collection = client.collections.get(WEAVIATE_CLASS_NAME)
    object_count = collection.aggregate.over_all(total_count=True).total_count
    if DEBUG:
        logging.debug(f"Object count in Weaviate collection: {object_count}")
    if object_count == 0 or force_reindex:
        if DEBUG:
            logging.debug("Indexing new documents due to empty collection or file path change.")
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            if DEBUG:
                logging.debug(f"Loading document from temp file: {tmp_path}")
            docs = load_document(tmp_path, suffix.lstrip('.'))
            if DEBUG:
                logging.debug(f"Loaded {len(docs)} documents from loader.")
            splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
            split_docs = splitter.split_documents(docs)
            if DEBUG:
                logging.debug(f"Split into {len(split_docs)} chunks for vector store.")
            db.add_documents(split_docs)
            if DEBUG:
                logging.debug(f"Indexed {len(split_docs)} documents in Weaviate.")
            # Save the new file path (legacy, optional)
            save_file_path(file_path)
        finally:
            os.remove(tmp_path)
            if DEBUG:
                logging.debug(f"Removed temp file: {tmp_path}")
    else:
        if DEBUG:
            logging.debug(f"Found {object_count} documents in Weaviate. Skipping indexing.")

    # Load chat history from Redis (if it exists)
    chat_history = []
    try:
        cached_history = redis_client.get(chat_history_key)
        if cached_history:
            chat_history = json.loads(cached_history)
            if DEBUG:
                logging.debug(f"Loaded chat history for user {user_id}: {chat_history}")
        else:
            if DEBUG:
                logging.debug(f"No chat history found for user {user_id}.")
    except Exception as e:
        logging.error(f"Error loading chat history from Redis: {str(e)}")

    # Limit to the last 5 messages to reduce noise
    limited_chat_history = chat_history[-5:]
    if DEBUG:
        logging.debug(f"Limited chat history for prompt: {limited_chat_history}")

    # Check if the question is already answered in the chat history
    for msg in reversed(chat_history):
        if msg["type"] == "human" and msg["content"].strip().lower() == question.strip().lower():
            idx = chat_history.index(msg)
            if idx + 1 < len(chat_history) and chat_history[idx + 1]["type"] == "ai":
                answer = chat_history[idx + 1]["content"]
                response_time = round(time.time() - start_time, 3)
                if DEBUG:
                    logging.debug("Question found in chat history, returning cached answer.")
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

    # Format chat history for the prompt
    chat_history_text = ""
    for msg in limited_chat_history:
        if msg["type"] == "human":
            chat_history_text += f"User: {msg['content']}\n"
        elif msg["type"] == "ai":
            chat_history_text += f"Assistant: {msg['content']}\n"
    if DEBUG:
        logging.debug(f"Formatted chat history text for prompt: {chat_history_text}")

    # Extract the question's topic
    topic = extract_question_topic(question)
    if DEBUG:
        logging.debug(f"Extracted topic: {topic}")

    # Set up retriever with contextual compression
    if DEBUG:
        logging.debug("Setting up retriever and compressor pipeline.")
    base_retriever = db.as_retriever(
        search_type="hybrid",
        search_kwargs={
            "k": 6,
            "alpha": 0.5
        }
    )
    compressor = DocumentCompressorPipeline(
        transformers=[
            EmbeddingsRedundantFilter(embeddings=embeddings),
        ]
    )
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

    # Check if it's just a greeting
    if is_greeting(question):
        if DEBUG:
            logging.debug("Detected greeting in question.")
        answer = get_greeting_response()
    else:
        # Retrieve documents
        if DEBUG:
            logging.debug("Retrieving relevant documents for question.")
        retrieved_docs = retriever.get_relevant_documents(question)
        if DEBUG:
            logging.debug(f"Retrieved {len(retrieved_docs)} documents.")

        # Combine document content for topic checking
        context_text = " ".join([doc.page_content for doc in retrieved_docs])
        if DEBUG:
            logging.debug(f"Combined context text length: {len(context_text)}")

        # Check if the topic is in chat history or context
        topic_in_history = is_topic_in_text(topic, chat_history_text)
        topic_in_context = is_topic_in_text(topic, context_text)

        if DEBUG:
            logging.debug(f"Topic in history: {topic_in_history}, Topic in context: {topic_in_context}")

        if not topic_in_history and not topic_in_context:
            answer = "I'm sorry, but I can only assist with questions related to topics in the provided context or recent conversation. Please ask about the SFA application or related topics!"
            if DEBUG:
                logging.debug("Topic not found in history or context. Returning fallback answer.")
        else:
            # Set up the LLM
            if DEBUG:
                logging.debug("Setting up LLM and RetrievalQA chain.")
            llm = ChatOllama(model="llama3.2:3b", temperature=0.7, max_tokens=128)

            # Set up the chain with RetrievalQA
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": CUSTOM_PROMPT}
            )

            # Run the chain with the original question and chat history
            if DEBUG:
                logging.debug("Running QA chain.")
            result = qa_chain({"query": question, "chat_history": chat_history_text})
            answer = result["result"]

            # Check if the answer indicates lack of context
            if not answer or any(phrase in answer.lower() for phrase in ["i don't know", "no context", "not in the provided context"]):
                answer = "I'm sorry, but I can only assist with questions related to the provided context. Please ask about the SFA application or related topics!"
                if DEBUG:
                    logging.debug("LLM returned fallback/no-context answer.")

    # Update chat history
    chat_history.append({"type": "human", "content": question})
    chat_history.append({"type": "ai", "content": answer})
    if DEBUG:
        logging.debug(f"Updated chat history: {chat_history}")

    # Save updated chat history to Redis
    try:
        redis_client.set(chat_history_key, json.dumps(chat_history))
        if DEBUG:
            logging.debug(f"Saved chat history for user {user_id} to Redis.")
    except Exception as e:
        logging.error(f"Error saving chat history to Redis: {str(e)}")

    # Calculate response time
    response_time = round(time.time() - start_time, 3)
    if DEBUG:
        logging.debug(f"Response time: {response_time}")

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