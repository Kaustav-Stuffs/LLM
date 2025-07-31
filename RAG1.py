import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

# ---- GLOBAL DEVICE SELECTION ----
def get_default_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
    

DEVICE = get_default_device()  # <-- Set your device globally here
print(f"Using device: {DEVICE}")
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
            top_k: int = 5,
            device: Optional[str] = None
    ):
        self.documents = documents
        self.texts = [doc['text'] for doc in documents]
        self.doc_ids = [doc['id'] for doc in documents]
        self.top_k = top_k
        total = vector_weight + bm25_weight
        self.vector_weight = vector_weight / total
        self.bm25_weight = bm25_weight / total

        # Use global DEVICE if device is not specified
        if device is None:
            device = DEVICE
        self.device = device

        print(f"Loading embedding model on device: {self.device} ...")
        self.embedding_model = SentenceTransformer(embedding_model_name, device=self.device)
        self.document_embeddings = self._create_document_embeddings()

        print("Building BM25 index...")
        self.bm25_tokenized_corpus = [text.lower().split() for text in self.texts]
        self.bm25 = BM25Okapi(self.bm25_tokenized_corpus)

    def _create_document_embeddings(self) -> np.ndarray:
        print("Creating document embeddings...")
        return self.embedding_model.encode(self.texts, show_progress_bar=True)

    def _vector_search(self, query: str, top_k: int = None) -> List[Tuple[int, float]]:
        if top_k is None:
            top_k = self.top_k
        query_embedding = self.embedding_model.encode(query)
        similarities = cosine_similarity([query_embedding], self.document_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(idx, similarities[idx]) for idx in top_indices]

    def _bm25_search(self, query: str, top_k: int = None) -> List[Tuple[int, float]]:
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

    def query_search(self, query: str, min_relevance_score: float = 0.5, reranker: Optional[callable] = None) -> List[Dict]:
        """
        Perform a search with the given query and return results as a list.

        Args:
            query: The search query
            min_relevance_score: Minimum relevance score to consider a result valid (0-1)
            reranker: Optional function to rerank results

        Returns:
            List of document dictionaries with added relevance scores
        """
        return self.search(query, reranker=reranker, min_relevance_score=min_relevance_score)

    def add_documents(self, new_documents: List[Dict]):
        if not new_documents:
            return
        self.documents.extend(new_documents)
        new_texts = [doc['text'] for doc in new_documents]
        self.texts.extend(new_texts)
        self.doc_ids.extend([doc['id'] for doc in new_documents])
        new_tokenized = [text.lower().split() for text in new_texts]
        self.bm25_tokenized_corpus.extend(new_tokenized)
        self.bm25 = BM25Okapi(self.bm25_tokenized_corpus)
        new_embeddings = self.embedding_model.encode(new_texts, show_progress_bar=True)
        self.document_embeddings = np.vstack([self.document_embeddings, new_embeddings])


# Load documents and initialize the search engine
documents_file = "./doc1.json"
try:
    with open(documents_file, 'r', encoding='utf-8') as f:
        documents = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"Error: File '{documents_file}' not found.")
except json.JSONDecodeError:
    raise ValueError(f"Error: File '{documents_file}' contains invalid JSON.")

# Create a single instance of HybridSearchRAG
search_engine = HybridSearchRAG(
    documents=documents,
    vector_weight=0.7,
    bm25_weight=0.3,
    top_k=3,
    device=DEVICE  # <-- Use the global device here
)