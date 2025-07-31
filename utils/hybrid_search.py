import numpy as np
from typing import List, Dict, Tuple, Optional
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

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
        
        # Normalize weights
        total = vector_weight + bm25_weight
        self.vector_weight = vector_weight / total
        self.bm25_weight = bm25_weight / total
        
        self.device = device or "cpu"
        
        logger.info(f"Initializing HybridSearchRAG with {len(documents)} documents on {self.device}")
        
        # Initialize TF-IDF vectorizer (faster alternative to sentence transformers)
        logger.info("Loading TF-IDF vectorizer...")
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Create document embeddings
        logger.info("Creating document embeddings...")
        self.document_embeddings = self._create_document_embeddings()
        
        # Initialize BM25
        logger.info("Building BM25 index...")
        self.bm25_tokenized_corpus = [text.lower().split() for text in self.texts]
        self.bm25 = BM25Okapi(self.bm25_tokenized_corpus)
        
        logger.info("HybridSearchRAG initialization complete")

    def _create_document_embeddings(self) -> np.ndarray:
        """Create TF-IDF embeddings for all documents."""
        return self.vectorizer.fit_transform(self.texts).toarray()

    def _vector_search(self, query: str, top_k: int = None) -> List[Tuple[int, float]]:
        """Perform vector similarity search."""
        if top_k is None:
            top_k = self.top_k
            
        query_embedding = self.vectorizer.transform([query]).toarray()[0]
        similarities = cosine_similarity([query_embedding], self.document_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(idx, similarities[idx]) for idx in top_indices]

    def _bm25_search(self, query: str, top_k: int = None) -> List[Tuple[int, float]]:
        """Perform BM25 lexical search."""
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
            return [(idx, 0.5) for idx, _ in results]
        
        score_range = max(scores) - min(scores)
        normalized = []
        
        for idx, score in results:
            normalized_score = (score - min(scores)) / score_range if score_range > 0 else 0.0
            normalized.append((idx, normalized_score))
            
        return normalized

    def search(
        self, 
        query: str, 
        min_relevance_score: float = 0.5,
        reranker: Optional[callable] = None
    ) -> List[Dict]:
        """
        Perform hybrid search combining vector and BM25 search.
        
        Args:
            query: Search query
            min_relevance_score: Minimum relevance score threshold
            reranker: Optional reranking function
            
        Returns:
            List of relevant documents with scores
        """
        if not query.strip():
            return []
        
        # Perform both types of search
        vector_results = self._vector_search(query, top_k=self.top_k * 2)
        bm25_results = self._bm25_search(query, top_k=self.top_k * 2)
        
        # Normalize scores
        vector_results = self._normalize_scores(vector_results)
        bm25_results = self._normalize_scores(bm25_results)
        
        # Combine scores
        combined_scores = {}
        
        # Add vector scores
        for idx, score in vector_results:
            combined_scores[idx] = self.vector_weight * score
        
        # Add BM25 scores
        for idx, score in bm25_results:
            if idx in combined_scores:
                combined_scores[idx] += self.bm25_weight * score
            else:
                combined_scores[idx] = self.bm25_weight * score
        
        # Sort by combined score and filter by relevance
        top_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:self.top_k]
        
        results = []
        for idx, score in top_results:
            if score >= min_relevance_score:
                doc = self.documents[idx].copy()
                doc['relevance_score'] = float(score)
                results.append(doc)
        
        # Apply reranking if provided
        if reranker and callable(reranker) and results:
            results = reranker(query, results)
        
        logger.info(f"Search for '{query[:50]}...' returned {len(results)} results")
        return results

    def query_search(
        self, 
        query: str, 
        min_relevance_score: float = 0.5, 
        reranker: Optional[callable] = None
    ) -> List[Dict]:
        """
        Perform a search with the given query and return results as a list.
        
        Args:
            query: The search query
            min_relevance_score: Minimum relevance score to consider a result valid (0-1)
            reranker: Optional function to rerank results
            
        Returns:
            List of document dictionaries with added relevance scores
        """
        return self.search(query, min_relevance_score=min_relevance_score, reranker=reranker)

    def add_documents(self, new_documents: List[Dict]):
        """Add new documents to the search index."""
        if not new_documents:
            return
            
        logger.info(f"Adding {len(new_documents)} new documents to index")
        
        # Update document store
        self.documents.extend(new_documents)
        new_texts = [doc['text'] for doc in new_documents]
        self.texts.extend(new_texts)
        self.doc_ids.extend([doc['id'] for doc in new_documents])
        
        # Update BM25 index
        new_tokenized = [text.lower().split() for text in new_texts]
        self.bm25_tokenized_corpus.extend(new_tokenized)
        self.bm25 = BM25Okapi(self.bm25_tokenized_corpus)
        
        # Update embeddings
        new_embeddings = self.vectorizer.transform(new_texts).toarray()
        self.document_embeddings = np.vstack([self.document_embeddings, new_embeddings])
        
        logger.info("Document addition complete")

    def get_stats(self) -> Dict:
        """Get statistics about the search index."""
        return {
            "total_documents": len(self.documents),
            "total_tokens": sum(len(tokens) for tokens in self.bm25_tokenized_corpus),
            "avg_document_length": np.mean([len(text) for text in self.texts]),
            "embedding_dimension": self.vectorizer.max_features,
            "device": self.device,
            "vector_weight": self.vector_weight,
            "bm25_weight": self.bm25_weight
        }