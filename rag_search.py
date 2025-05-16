from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

class HybridSearchRAG:
    """
    Hybrid Search implementation for Retrieval Augmented Generation.
    Combines vector-based semantic search with BM25 lexical search.
    """
    def __init__(
            self,
            documents: List[Dict],
            embedding_model_name: str = "all-MiniLM-L6-v2",
            vector_weight: float = 0.6,
            bm25_weight: float = 0.4,
            top_k: int = 3
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
        return [(idx, score) for idx, score in zip(top_indices, bm25_scores[top_indices])]

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

# from typing import List, Dict, Optional
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from rank_bm25 import BM25Okapi
# from sklearn.metrics.pairwise import cosine_similarity

# class HybridSearchRAG:
#     """
#     Hybrid Search implementation for Retrieval Augmented Generation.
#     Combines vector-based semantic search with BM25 lexical search.
#     """
#     def __init__(
#             self,
#             documents: List[Dict],
#             embedding_model_name: str = "all-MiniLM-L6-v2",
#             vector_weight: float = 0.7,
#             bm25_weight: float = 0.3,
#             top_k: int = 5
#     ):
#         self.documents = documents
#         self.texts = [doc['text'] for doc in documents]
#         self.doc_ids = [doc['id'] for doc in documents]
#         self.top_k = top_k

#         # Initialize weights (will be adjusted dynamically)
#         self.vector_weight = vector_weight
#         self.bm25_weight = bm25_weight
#         self._normalize_weights()

#         # Initialize vector search
#         print("Loading embedding model...")
#         self.embedding_model = SentenceTransformer(embedding_model_name)
#         self.document_embeddings = self._create_document_embeddings()

#         # Initialize BM25 search
#         print("Building BM25 index...")
#         self.bm25_tokenized_corpus = [text.lower().split() for text in self.texts]
#         self.bm25 = BM25Okapi(self.bm25_tokenized_corpus)

#     def _normalize_weights(self):
#         """Ensure weights sum to 1."""
#         total = self.vector_weight + self.bm25_weight
#         self.vector_weight = self.vector_weight / total
#         self.bm25_weight = self.bm25_weight / total

#     def adjust_weights(self, query: str):
#         """Dynamically adjust vector and BM25 weights based on query length."""
#         query_length = len(query.split())
#         if query_length <= 5: 
#             self.vector_weight = 0.3
#             self.bm25_weight = 0.7
#         else:  # Long queries: prioritize vector
#             self.vector_weight = 0.7
#             self.bm25_weight = 0.3
#         self._normalize_weights()
#         print(f"[DEBUG] Adjusted weights - vector: {self.vector_weight:.2f}, BM25: {self.bm25_weight:.2f} for query length: {query_length}")

#     def _create_document_embeddings(self) -> np.ndarray:
#         print("Creating document embeddings...")
#         return self.embedding_model.encode(self.texts, show_progress_bar=True)

#     def _vector_search(self, query: str, top_k: int = None) -> List[tuple[int, float]]:
#         if top_k is None:
#             top_k = self.top_k
#         query_embedding = self.embedding_model.encode(query)
#         similarities = cosine_similarity([query_embedding], self.document_embeddings)[0]
#         top_indices = np.argsort(similarities)[::-1][:top_k]
#         return [(idx, similarities[idx]) for idx in top_indices]

#     def _bm25_search(self, query: str, top_k: int = None) -> List[tuple[int, float]]:
#         if top_k is None:
#             top_k = self.top_k
#         tokenized_query = query.lower().split()
#         bm25_scores = self.bm25.get_scores(tokenized_query)
#         top_indices = np.argsort(bm25_scores)[::-1][:top_k]
#         return [(idx, score) for idx, score in zip(top_indices, bm25_scores[top_indices])]

#     def search(self, query: str, reranker: Optional[callable] = None, min_relevance_score: float = 0.5) -> List[Dict]:
#         # Adjust weights based on query
#         self.adjust_weights(query)

#         query_keywords = set(query.lower().split()) - set(['is', 'are', 'there', 'any', 'in', 'this', 'the', 'a', 'an'])
#         has_keyword_match = False
#         for text in self.texts:
#             text_lower = text.lower()
#             if any(keyword in text_lower for keyword in query_keywords):
#                 has_keyword_match = True
#                 break
#         if not has_keyword_match:
#             return []

#         vector_results = self._vector_search(query, top_k=self.top_k * 2)
#         bm25_results = self._bm25_search(query, top_k=self.top_k * 2)

#         def normalize_scores(results):
#             scores = [score for _, score in results]
#             if not scores or max(scores) == min(scores):
#                 return [(idx, 0.0) for idx, _ in results]
#             score_range = max(scores) - min(scores)
#             return [(idx, (score - min(scores)) / score_range if score_range > 0 else 0.0)
#                     for idx, score in results]

#         vector_results = normalize_scores(vector_results)
#         bm25_results = normalize_scores(bm25_results)

#         combined_scores = {}
#         for idx, score in vector_results:
#             combined_scores[idx] = self.vector_weight * score
#         for idx, score in bm25_results:
#             if idx in combined_scores:
#                 combined_scores[idx] += self.bm25_weight * score
#             else:
#                 combined_scores[idx] = self.bm25_weight * score

#         top_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:self.top_k]
#         results = []
#         for idx, score in top_results:
#             if score >= min_relevance_score:
#                 doc = self.documents[idx].copy()
#                 doc['relevance_score'] = float(score)
#                 results.append(doc)

#         if reranker and callable(reranker) and results:
#             results = reranker(query, results)
#         return results