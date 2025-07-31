import logging
import re
from typing import Dict, List

logger = logging.getLogger(__name__)

class QueryProcessor:
    """
    Processes user queries and generates responses using RAG and LLM.
    """
    
    def __init__(self):
        # Simple and reliable greeting patterns
        self.greeting_patterns = [
            r"^hi+\b",  # hi, hii, hiii
            r"^hey+\b",  # hey, heyy
            r"^hello+\b",  # hello, helloo
            r"^good\s*(morning|afternoon|evening|day)\b",
            r"^greetings?\b",
            r"^howdy\b",
            r"^yo+\b",
            r"^sup\b",
            r"^what'?s?\s*up\b",
            r"^how\s*are\s*you\b"
        ]
        
        # Simple closing patterns
        self.closing_patterns = [
            r"^thank\s*you\b", r"^thanks\b", r"^thx\b", r"^ty\b",
            r"^bye+\b", r"^goodbye\b", r"^see\s*you\b", r"^see\s*ya\b",
            r"^ok\b", r"^okay\b", r"^got\s*it\b", r"^understood\b",
            r"^good\s*bye\b", r"^take\s*care\b",
            r"^peace\b", r"^later\b", r"^done\b", r"^finished\b"
        ]
        
        # Common simple queries that don't need LLM processing
        self.simple_responses = {
            "hi": "Hello! I'm ready to help you with questions about your uploaded documents. What would you like to know?",
            "hello": "Hi there! I can answer questions based on the documents you've uploaded. How can I help?",
            "hey": "Hey! I'm here to help you explore your documents. What questions do you have?",
            "thanks": "You're welcome! Feel free to ask more questions about your documents.",
            "thank you": "You're welcome! I'm here to help with any questions about your uploaded content.",
            "bye": "Goodbye! I'll be here whenever you need help with your documents.",
            "goodbye": "Take care! I'm always ready to help you explore your documents."
        }
        
        logger.info("QueryProcessor initialized with smart greeting detection")
    
    def process_query(
        self,
        query: str,
        search_engine,
        llm_handler,
        model_choice: str,
        min_relevance_score: float = 0.5
    ) -> Dict:
        """
        Process a user query and generate a response.
        
        Args:
            query: User's question
            search_engine: HybridSearchRAG instance
            llm_handler: LLMHandler instance
            model_choice: Either 'gemini' or 'ollama'
            min_relevance_score: Minimum relevance threshold
            
        Returns:
            Dict with answer, relevance scores, and matched documents
        """
        # Quick check for simple queries first (no logging for efficiency)
        simple_response = self._get_simple_response(query)
        if simple_response:
            logger.debug(f"Quick response for simple query: '{query}'")
            return {
                "answer": simple_response,
                "relevance_scores": [],
                "matched_documents": []
            }
        
        logger.info(f"Processing complex query: '{query[:50]}...' with model: {model_choice}")
        
        # Handle greetings (fallback for complex greetings)
        if self._is_greeting_only(query):
            return {
                "answer": "Hello! I'm ready to help you with questions about your uploaded documents. What would you like to know?",
                "relevance_scores": [],
                "matched_documents": []
            }
        
        # Handle closings (fallback for complex closings)
        if self._is_closing_only(query):
            return {
                "answer": "You're welcome! Feel free to upload more documents or ask more questions anytime.",
                "relevance_scores": [],
                "matched_documents": []
            }
        
        # Perform search
        try:
            search_results = search_engine.query_search(
                query, 
                min_relevance_score=min_relevance_score
            )
            
            if not search_results:
                return {
                    "answer": "I couldn't find relevant information in your uploaded documents to answer this question. Please make sure your question relates to the content you've uploaded.",
                    "relevance_scores": [],
                    "matched_documents": []
                }
            
            logger.info(f"Found {len(search_results)} relevant documents")
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return {
                "answer": "Sorry, there was an error searching through your documents.",
                "relevance_scores": [],
                "matched_documents": []
            }
        
        # Prepare context
        combined_context = "\n\n".join([result['text'] for result in search_results])
        
        # Generate response
        try:
            response = self._generate_response(query, combined_context, llm_handler, model_choice)
            
            return {
                "answer": response,
                "relevance_scores": [result['relevance_score'] for result in search_results],
                "matched_documents": [result['text'] for result in search_results]
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "answer": "Sorry, there was an error generating a response to your question.",
                "relevance_scores": [result['relevance_score'] for result in search_results],
                "matched_documents": [result['text'] for result in search_results]
            }
    
    def _generate_response(self, query: str, context: str, llm_handler, model_choice: str) -> str:
        """Generate response using LLM."""
        
        # Add greeting if query starts with one
        greeting_prefix = ""
        if self._query_starts_with_greeting(query):
            greeting_prefix = "Hello! "
        
        # Prepare prompt
        prompt = self._create_prompt(query, context)
        
        # Generate response
        response = llm_handler.generate_response(prompt)
        
        # Add greeting prefix if needed
        if greeting_prefix and not re.match(r"^\s*hello[!.,\s]*", response, re.IGNORECASE):
            response = greeting_prefix + response
        
        return response
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create a well-structured prompt for the LLM."""
        return f"""You are a helpful AI assistant that answers questions based strictly on provided document content.

IMPORTANT INSTRUCTIONS:
- Answer ONLY using information from the provided context
- Do NOT use external knowledge or make assumptions
- If the context doesn't contain enough information to answer the question, say so clearly
- Keep your response concise and directly relevant to the question
- Do not mention or repeat the user's question in your response
- Do not reference "the document" or "the context" in your response

CONTEXT FROM UPLOADED DOCUMENTS:
{context}

USER QUESTION: {query}

RESPONSE:"""
    
    def _get_simple_response(self, query: str) -> str:
        """Get instant response for simple queries without LLM processing."""
        q = query.lower().strip()
        q = re.sub(r'[^\w\s]', '', q)  # Remove punctuation
        
        # Direct matches for common simple queries
        if q in self.simple_responses:
            return self.simple_responses[q]
        
        # Check for greeting patterns (very fast)
        for pattern in self.greeting_patterns:
            if re.match(pattern, q):
                # Only if it's a simple greeting (1-2 words max)
                words = q.split()
                if len(words) <= 2:
                    return self.simple_responses.get("hi", "Hello! I'm ready to help you with questions about your uploaded documents. What would you like to know?")
        
        # Check for closing patterns (very fast)
        for pattern in self.closing_patterns:
            if re.match(pattern, q):
                words = q.split()
                if len(words) <= 2:
                    return self.simple_responses.get("bye", "Goodbye! I'll be here whenever you need help with your documents.")
        
        return None  # No simple response found, needs full processing

    def _is_greeting_only(self, query: str) -> bool:
        """Check if query is only a greeting (fallback for complex greetings)."""
        q = query.lower().strip()
        q = re.sub(r'[^\w\s]', '', q)  # Remove punctuation
        words = q.split()
        
        if len(words) == 0:
            return False
        
        # Simple check for greeting patterns
        first_word = words[0]
        greeting_words = {'hi', 'hello', 'hey', 'greetings', 'howdy', 'yo', 'sup'}
        
        if first_word in greeting_words and len(words) <= 3:
            # Check if remaining words are not content-related
            if len(words) == 1:
                return True
            
            remaining_words = words[1:]
            content_words = {'what', 'how', 'why', 'when', 'where', 'who', 'can', 'will', 'should',
                            'document', 'file', 'text', 'question', 'answer', 'help', 'about', 'tell'}
            
            # If any remaining word suggests content query, it's not just a greeting
            has_content = any(word in content_words for word in remaining_words)
            return not has_content
        
        return False
    
    def _is_closing_only(self, query: str) -> bool:
        """Check if query is only a closing."""
        q = query.lower().strip()
        q = re.sub(r'[^\w\s]', '', q)
        words = q.split()
        
        if len(words) == 0:
            return False
            
        # Check if the entire query matches closing patterns
        is_closing = any(re.match(pattern, q) for pattern in self.closing_patterns)
        
        # Also check for standalone closing words
        standalone_closings = {'ok', 'okay', 'thanks', 'thank', 'bye', 'goodbye', 'done', 'ty', 'thx'}
        
        return is_closing or (len(words) <= 2 and any(word in standalone_closings for word in words))
    
    def _query_starts_with_greeting(self, query: str) -> bool:
        """Check if query starts with a greeting but has additional content."""
        q = query.lower().strip()
        q = re.sub(r'[^\w\s]', '', q)
        words = q.split()
        
        if len(words) == 0:
            return False
            
        # Check first 1-2 words for greeting patterns
        first_words = ' '.join(words[:2]) if len(words) > 1 else words[0]
        starts_with_greeting = any(re.match(pattern, first_words) for pattern in self.greeting_patterns)
        
        # Only return True if it starts with greeting AND has additional meaningful content
        if starts_with_greeting and len(words) > 2:
            # Check if there are content words after the greeting
            remaining_words = words[2:]
            content_words = {'what', 'how', 'why', 'when', 'where', 'who', 'can', 'will', 'should', 
                            'document', 'file', 'text', 'question', 'answer', 'help', 'about', 'tell'}
            has_content = any(word in content_words for word in remaining_words) or len(remaining_words) > 2
            return has_content
            
        return False