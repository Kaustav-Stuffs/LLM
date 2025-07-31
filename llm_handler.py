import os
import logging
from typing import Optional

# Gemini imports
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Ollama imports - disabled for offline mode
OLLAMA_AVAILABLE = False

from greeting_handler import GreetingHandler

logger = logging.getLogger(__name__)

class LLMHandler:
    """Handles interactions with different LLM providers."""
    
    def __init__(self):
        self.gemini_client = None
        self.greeting_handler = GreetingHandler()
        
        # Initialize Gemini if available
        if GEMINI_AVAILABLE:
            self._init_gemini()
        
    def _init_gemini(self):
        """Initialize Gemini client."""
        try:
            api_key = os.getenv("AIzaSyDxYogQnjYmTbvghsqSWCKtot5DpygC8hA")
            if api_key:
                self.gemini_client = genai.Client(api_key=api_key)
                logger.info("Gemini client initialized successfully")
            else:
                logger.warning("GEMINI_API_KEY not found in environment variables")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {str(e)}")
            self.gemini_client = None
    
    def _generate_fallback_response(self, query: str, context: str) -> str:
        """Generate a simple fallback response when no LLM is available."""
        if context.strip():
            return f"Based on the provided documents, here's what I found:\n\n{context[:500]}{'...' if len(context) > 500 else ''}\n\nPlease note: This is a direct excerpt from your documents. For AI-powered analysis, ensure your Gemini API key is properly configured."
        else:
            return "I couldn't find relevant information in the uploaded documents to answer your question."
    
    def generate_answer(
        self,
        query: str,
        context: str,
        model_choice: str = "gemini",
        ollama_model: Optional[str] = None
    ) -> str:
        """
        Generate an answer using the specified model with intelligent greeting/closing handling.
        
        Args:
            query: User's question
            context: Relevant context from documents
            model_choice: Either "gemini" or "ollama"
            ollama_model: Specific Ollama model name if using Ollama
            
        Returns:
            Generated answer string
        """
        # Handle greetings only
        if self.greeting_handler.is_greeting_only(query):
            return self.greeting_handler.get_greeting_response()
        
        # Handle closings only
        if self.greeting_handler.is_closing_only(query):
            return self.greeting_handler.get_closing_response()
        
        # For regular queries, generate response and add greeting prefix if needed
        if not context.strip():
            response = "I couldn't find any relevant information in the uploaded documents to answer your question. Please make sure you've uploaded documents and try asking a more specific question."
        else:
            prompt = self._create_prompt(query, context)
            
            if model_choice == "gemini":
                response = self._generate_with_gemini(prompt)
            else:
                response = self._generate_fallback_response(query, context)
        
        # Add greeting prefix if user started with a greeting
        return self.greeting_handler.should_add_greeting_prefix(query, response)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create a prompt for the LLM."""
        return f"""You are a helpful AI assistant that answers questions based on provided context.

Instructions:
- Answer the question using ONLY the information provided in the context below
- If the context doesn't contain enough information to answer the question, say so clearly
- Be concise and accurate in your response
- Do not make up or assume information not present in the context
- If asked about something not covered in the context, explain that the information is not available

Context:
{context}

Question: {query}

Answer:"""
    
    def _generate_with_gemini(self, prompt: str) -> str:
        """Generate response using Gemini."""
        if not self.gemini_client:
            return "Gemini is not available. Please check your API key configuration."
        
        try:
            response = self.gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            
            if response and response.text:
                return response.text.strip()
            else:
                return "Sorry, I couldn't generate a response. Please try again."
                
        except Exception as e:
            logger.error(f"Error generating response with Gemini: {str(e)}")
            return f"Error generating response: {str(e)}"
    

    
    def is_gemini_available(self) -> bool:
        """Check if Gemini is available and configured."""
        return self.gemini_client is not None
    
    def get_available_models(self) -> dict:
        """Get information about available models."""
        return {
            "gemini": {
                "available": self.is_gemini_available(),
                "models": ["gemini-2.5-flash", "gemini-2.5-pro"] if self.is_gemini_available() else []
            }
        }
