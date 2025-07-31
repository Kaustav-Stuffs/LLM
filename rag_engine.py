import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import logging

logger = logging.getLogger(__name__)

def get_default_device():
    """Determine the best available device for computation."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

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
        self.doc_ids = [doc.get('id', i) for i, doc in enumerate(documents)]
        self.top_k = top_k
        
        # Normalize weights
        total = vector_weight + bm25_weight
        self.vector_weight = vector_weight / total
        self.bm25_weight = bm25_weight / total

        # Device setup
        if device is None:
            device = get_default_device()
        self.device = device

        logger.info(f"Loading embedding model on device: {self.device}")
        self.embedding_model = SentenceTransformer(embedding_model_name, device=self.device)
        self.document_embeddings = self._create_document_embeddings()

        logger.info("Building BM25 index...")
        self.bm25_tokenized_corpus = [text.lower().split() for text in self.texts]
        self.bm25 = BM25Okapi(self.bm25_tokenized_corpus)

    def _create_document_embeddings(self) -> np.ndarray:
        """Create embeddings for all documents."""
        logger.info("Creating document embeddings...")
        return self.embedding_model.encode(self.texts, show_progress_bar=False)

    def _vector_search(self, query: str, top_k: int = None) -> List[Tuple[int, float]]:
        """Perform semantic vector search."""
        if top_k is None:
            top_k = self.top_k
        
        query_embedding = self.embedding_model.encode(query)
        similarities = cosine_similarity([query_embedding], self.document_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(idx, similarities[idx]) for idx in top_indices]

    def _bm25_search(self, query: str, top_k: int = None) -> List[Tuple[int, float]]:
        """Perform lexical BM25 search."""
        if top_k is None:
            top_k = self.top_k
        
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        return [(idx, bm25_scores[idx]) for idx in top_indices]

    def _normalize_scores(self, results: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        """Normalize scores to 0-1 range."""
        if not results:
            return results
        
        scores = [score for _, score in results]
        if max(scores) == min(scores):
            return [(idx, 0.0) for idx, _ in results]
        
        score_range = max(scores) - min(scores)
        return [(idx, (score - min(scores)) / score_range if score_range > 0 else 0.0)
                for idx, score in results]

    def search(self, query: str, min_relevance_score: float = 0.5, reranker: Optional[callable] = None) -> List[Dict]:
        """
        Perform hybrid search combining vector and BM25 results.
        
        Args:
            query: Search query string
            min_relevance_score: Minimum relevance score threshold
            reranker: Optional function to rerank results
            
        Returns:
            List of relevant documents with relevance scores
        """
        # Perform both searches
        vector_results = self._vector_search(query, top_k=self.top_k * 2)
        bm25_results = self._bm25_search(query, top_k=self.top_k * 2)

        # Normalize scores
        vector_results = self._normalize_scores(vector_results)
        bm25_results = self._normalize_scores(bm25_results)

        # Combine scores
        combined_scores = {}
        for idx, score in vector_results:
            combined_scores[idx] = self.vector_weight * score
        
        for idx, score in bm25_results:
            if idx in combined_scores:
                combined_scores[idx] += self.bm25_weight * score
            else:
                combined_scores[idx] = self.bm25_weight * score

        # Get top results
        top_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:self.top_k]
        
        # Filter by minimum relevance and format results
        results = []
        for idx, score in top_results:
            if score >= min_relevance_score:
                doc = self.documents[idx].copy()
                doc['relevance_score'] = float(score)
                results.append(doc)

        # Apply reranker if provided
        if reranker and callable(reranker) and results:
            results = reranker(query, results)

        return results

    def add_documents(self, new_documents: List[Dict]):
        """Add new documents to the search index."""
        if not new_documents:
            return
        
        self.documents.extend(new_documents)
        new_texts = [doc['text'] for doc in new_documents]
        self.texts.extend(new_texts)
        self.doc_ids.extend([doc.get('id', len(self.doc_ids) + i) for i, doc in enumerate(new_documents)])
        
        # Update BM25 index
        new_tokenized = [text.lower().split() for text in new_texts]
        self.bm25_tokenized_corpus.extend(new_tokenized)
        self.bm25 = BM25Okapi(self.bm25_tokenized_corpus)
        
        # Update embeddings
        new_embeddings = self.embedding_model.encode(new_texts, show_progress_bar=False)
        self.document_embeddings = np.vstack([self.document_embeddings, new_embeddings])

    def get_stats(self) -> Dict:
        """Get statistics about the search engine."""
        return {
            "total_documents": len(self.documents),
            "embedding_dimension": self.document_embeddings.shape[1] if self.document_embeddings is not None else 0,
            "device": self.device,
            "model_name": self.embedding_model.get_sentence_embedding_dimension() if hasattr(self.embedding_model, 'get_sentence_embedding_dimension') else "unknown"
        }
