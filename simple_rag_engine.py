import json
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from collections import Counter
import re

logger = logging.getLogger(__name__)

class SimpleRAG:
    """
    Simple RAG implementation using only BM25-like lexical search without embeddings.
    This is a fallback when sentence-transformers is not available.
    """

    def __init__(
        self,
        documents: List[Dict],
        top_k: int = 5
    ):
        self.documents = documents
        self.texts = [doc['text'] for doc in documents]
        self.doc_ids = [doc.get('id', i) for i, doc in enumerate(documents)]
        self.top_k = top_k
        
        logger.info("Building simple lexical search index...")
        self.word_freq = self._build_word_frequency()
        self.doc_word_freq = self._build_document_word_frequency()

    def _build_word_frequency(self) -> Dict[str, int]:
        """Build global word frequency for the corpus."""
        word_freq = Counter()
        for text in self.texts:
            words = self._tokenize(text)
            word_freq.update(words)
        return dict(word_freq)

    def _build_document_word_frequency(self) -> List[Dict[str, int]]:
        """Build word frequency for each document."""
        doc_word_freq = []
        for text in self.texts:
            words = self._tokenize(text)
            doc_word_freq.append(Counter(words))
        return doc_word_freq

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Convert to lowercase and extract words
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        # Filter out very short words and common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we',
            'they', 'me', 'him', 'her', 'us', 'them'
        }
        return [word for word in words if len(word) >= 3 and word not in stop_words]

    def _calculate_tf_idf_score(self, query_words: List[str], doc_idx: int) -> float:
        """Calculate a simple TF-IDF-like score."""
        if doc_idx >= len(self.doc_word_freq):
            return 0.0
        
        doc_word_freq = self.doc_word_freq[doc_idx]
        doc_length = sum(doc_word_freq.values())
        
        if doc_length == 0:
            return 0.0
        
        score = 0.0
        for word in query_words:
            if word in doc_word_freq:
                # Term frequency
                tf = doc_word_freq[word] / doc_length
                
                # Inverse document frequency (simplified)
                docs_with_word = sum(1 for doc_freq in self.doc_word_freq if word in doc_freq)
                if docs_with_word > 0:
                    idf = np.log(len(self.documents) / docs_with_word)
                else:
                    idf = 0
                
                score += tf * idf
        
        return score

    def search(self, query: str, min_relevance_score: float = 0.1) -> List[Dict]:
        """
        Perform simple lexical search.
        
        Args:
            query: Search query string
            min_relevance_score: Minimum relevance score threshold
            
        Returns:
            List of relevant documents with relevance scores
        """
        query_words = self._tokenize(query)
        if not query_words:
            return []
        
        # Calculate scores for all documents
        scores = []
        for i in range(len(self.documents)):
            score = self._calculate_tf_idf_score(query_words, i)
            scores.append((i, score))
        
        # Sort by score and get top results
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by minimum relevance and format results
        results = []
        for idx, score in scores[:self.top_k]:
            if score >= min_relevance_score:
                doc = self.documents[idx].copy()
                doc['relevance_score'] = float(score)
                results.append(doc)
        
        return results

    def add_documents(self, new_documents: List[Dict]):
        """Add new documents to the search index."""
        if not new_documents:
            return
        
        self.documents.extend(new_documents)
        new_texts = [doc['text'] for doc in new_documents]
        self.texts.extend(new_texts)
        self.doc_ids.extend([doc.get('id', len(self.doc_ids) + i) for i, doc in enumerate(new_documents)])
        
        # Rebuild indices
        self.word_freq = self._build_word_frequency()
        self.doc_word_freq = self._build_document_word_frequency()

    def get_stats(self) -> Dict:
        """Get statistics about the search engine."""
        return {
            "total_documents": len(self.documents),
            "total_unique_words": len(self.word_freq),
            "search_type": "simple_lexical"
        }