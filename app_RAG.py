import streamlit as st
import json
import csv
import logging
from webcamaccess import process_query
from langchain_ollama import ChatOllama
import google.generativeai as genai

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Gemini Setup
genai.configure(api_key="AIzaSyDxYogQnjYmTbvghsqSWCKtot5DpygC8hA")
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# Qwen2.5 Setup
qwen_model = ChatOllama(model="llama3.2:3b", temperature=0.1, max_tokens=128)

def extract_context_from_files(file_paths):
    context_parts = []
    for path in file_paths:
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
        except Exception as e:
            logger.error(f"Error reading file {path}: {str(e)}")
    return "\n".join(context_parts)

DOCUMENT_PATHS = ["./doc1.json"]
CONTEXT = extract_context_from_files(DOCUMENT_PATHS)

st.title("RAG Chatbot Demo")

user_id = st.number_input("User ID", min_value=1, value=1)
question = st.text_area("Ask your question:")
model_choice = st.selectbox("Choose Model", ["llama", "gemini"])

if st.button("Get Answer"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            result = process_query(
                question,
                min_relevance_score=0.5,
                model_choice=model_choice,
                gemini_model=gemini_model,
                qwen_model=qwen_model
            )
            st.success("Answer:")
            st.write(result["answer"])
            st.info(f"Model used: {model_choice}")