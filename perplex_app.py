import streamlit as st
import os, json, csv, logging
from typing import List, Dict
from threading import Lock

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import numpy as np

# ====== Logging ======
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====== Simplified Document Preprocessing ======
def parse_uploaded_file(uploaded_file) -> List[Dict]:
    """Accepts a file-like object, returns structured text documents with metadata."""
    docs = []
    filename = uploaded_file.name
    filetype = filename.split('.')[-1]
    try:
        content = uploaded_file.read()
        if filetype == "json":
            data = json.loads(content.decode("utf-8"))
            if isinstance(data, list):
                for i, entry in enumerate(data):
                    docs.append({
                        "id": f"{filename}-{i}",
                        "text": entry["text"] if "text" in entry else str(entry),
                        "metadata": entry.get("metadata", {})
                    })
            elif isinstance(data, dict):
                text = data.get("text", json.dumps(data))
                docs.append({"id": f"{filename}-0", "text": text, "metadata": data.get("metadata", {})})
        elif filetype == "csv":
            rows = list(csv.reader(content.decode("utf-8").splitlines()))
            text = "\n".join([", ".join(row) for row in rows])
            docs.append({"id": filename, "text": text, "metadata": {}})
        elif filetype == "txt":
            text = content.decode("utf-8")
            docs.append({"id": filename, "text": text, "metadata": {}})
        else:
            st.warning("Unsupported file type, skipping.")
    except Exception as e:
        logger.warning(f"Parse error: {e}")
    return docs

# ====== Hybrid Search (vector + bm25) ======
class HybridSearchEngine:
    def __init__(self, docs: List[Dict], model_name="all-MiniLM-L6-v2", vector_weight=0.7, bm25_weight=0.3, top_k=3):
        self.docs = docs
        self.texts = [d['text'] for d in docs]
        self.doc_ids = [d['id'] for d in docs]
        self.top_k = top_k
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.embedder = SentenceTransformer(model_name)
        self.embed_matrix = self.embedder.encode(self.texts, show_progress_bar=True)
        self.bm25 = BM25Okapi([text.lower().split() for text in self.texts])

    def search(self, query, min_score=0.45):
        # Vector Search
        q_emb = self.embedder.encode([query])[0]
        vec_scores = cosine_similarity([q_emb], self.embed_matrix)[0]
        vec_pairs = list(enumerate(vec_scores))
        # BM25
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_pairs = list(enumerate(bm25_scores))
        # Normalize scores
        def norm(lst):
            arr = np.array([x[1] for x in lst])
            if arr.max() - arr.min() < 1e-6:
                arr[:] = 0
            else:
                arr = (arr - arr.min()) / (arr.max() - arr.min())
            return {i: float(score) for (i, _), score in zip(lst, arr)}
        v = norm(vec_pairs)
        b = norm(bm25_pairs)
        # Combine and fetch top
        scores = {}
        for i in range(len(self.texts)):
            scores[i] = self.vector_weight * v.get(i, 0) + self.bm25_weight * b.get(i, 0)
        sorted_ids = sorted(scores, key=lambda k: -scores[k])
        results = []
        for i in sorted_ids[:self.top_k]:
            if scores[i] >= min_score:
                d = self.docs[i].copy()
                d['relevance_score'] = float(scores[i])
                results.append(d)
        return results

# ====== Model Completion (You can plug LLM here; demo: stub) ======
def strict_answer_from_context(context, query):
    if not context.strip():
        return "I cannot answer this question. Please ask a specific question about the document."
    # This is where you'd use LLM like Gemini/Ollama with the context,
    # but you MUST instruct the LLM: "Only use the Document Context below. No assumptions. No info not in context."
    # For demo, we just echo the first line of context.
    return context.split('\n')[0].strip() if context else "I cannot answer this question. Please ask a specific question."


# ====== UI ======
st.set_page_config(page_title="General-Purpose RAG Chatbot", page_icon="🤖", layout="centered")
st.title("General-Purpose Retrieval Augmented Generation (RAG) Chatbot")
st.write("Upload a source file (txt, csv, or JSON with field 'text'), then ask your question. All answers are strictly from your uploaded document!")

uploaded_file = st.file_uploader("📎 Upload your document (txt, csv, or JSON)", type=["txt", "csv", "json"])
if uploaded_file:
    # --- Parse and search setup ---
    with st.spinner("Processing and indexing the document..."):
        parsed_docs = parse_uploaded_file(uploaded_file)
        if not parsed_docs:
            st.error("Could not parse any valid document content. Try another file.")
            st.stop()
        hybrid_search = HybridSearchEngine(parsed_docs)
        st.success("Document indexed! Ask your question below:")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # --- Chat interface ---
    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["user"])
        with st.chat_message("assistant"):
            st.write(turn["bot"])

    user_q = st.chat_input("Type your question...")
    if user_q:
        # Hybrid retrieval
        search_results = hybrid_search.search(user_q)
        context = "\n".join([r['text'] for r in search_results])
        answer = strict_answer_from_context(context, user_q)
        # Store in history and display
        st.session_state.chat_history.append({"user": user_q, "bot": answer})
        with st.chat_message("user"):
            st.write(user_q)
        with st.chat_message("assistant"):
            st.write(answer)

else:
    st.info("Please upload a document to begin.")

