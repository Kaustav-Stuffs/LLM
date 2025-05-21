import streamlit as st
import json
import csv
import logging
import hashlib
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

st.set_page_config(page_title="Customer Support Chatbot", page_icon="💬", layout="wide")

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=80)
    st.title("Support Chatbot")
    if "user_name" not in st.session_state:
        st.session_state.user_name = ""
    user_name = st.text_input("👤 Your Name", value=st.session_state.user_name)
    if user_name and st.session_state.user_name == "":
        st.session_state.user_name = user_name
    model_choice = st.selectbox("🤖 Choose Model", ["llama", "gemini"])
    if st.button("🧹 Clear chat"):
        st.session_state.chat_history = []

# --- MAIN CHAT ---
st.markdown(
    """
    <style>
    .user-bubble {
        background-color: #DCF8C6;
        color: #222;
        padding: 10px 16px;
        border-radius: 18px 18px 2px 18px;
        margin-bottom: 8px;
        max-width: 70%;
        float: right;
        clear: both;
    }
    .bot-bubble {
        background-color: #F1F0F0;
        color: #222;
        padding: 10px 16px;
        border-radius: 18px 18px 18px 2px;
        margin-bottom: 8px;
        max-width: 70%;
        float: left;
        clear: both;
    }
    .avatar {
        width: 32px;
        vertical-align: middle;
        margin-right: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("💬 Customer Support Chat")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def get_user_id(name: str) -> int:
    return int(hashlib.sha256(name.encode()).hexdigest(), 16) % 10**8

def render_chat():
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-bubble"><img src="https://cdn-icons-png.flaticon.com/512/1946/1946429.png" class="avatar"/><b>{st.session_state.user_name}:</b> {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="bot-bubble"><img src="https://cdn-icons-png.flaticon.com/512/4712/4712035.png" class="avatar"/><b>Support Bot:</b> {msg["content"]}</div>',
                unsafe_allow_html=True,
            )

if st.session_state.user_name:
    render_chat()
    prompt = st.text_input(
        "Type your message and press Enter...",
        key="input",
        placeholder="How can I help you today?"
    )
    send = st.button("Send", use_container_width=True)
    if (send or (prompt and st.session_state.get("last_prompt") != prompt)):
        if prompt:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.session_state.last_prompt = prompt
            user_id = get_user_id(st.session_state.user_name)
            with st.spinner("Bot is typing..."):
                result = process_query(
                    prompt,
                    min_relevance_score=0.5,
                    model_choice=model_choice,
                    gemini_model=gemini_model,
                    qwen_model=qwen_model
                )
                answer = result["answer"]
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.experimental_rerun()  # This will clear the input box
else:
    st.info("Please enter your name in the sidebar to start chatting.")