import os
import logging
from typing import Optional, List, Dict
import subprocess
import json

# Gemini imports
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Ollama imports
try:
    from langchain_ollama import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

logger = logging.getLogger(__name__)

class LLMHandler:
    """
    Handles both online (Gemini) and offline (Ollama) language models.
    """
    
    def __init__(self):
        self.gemini_client = None
        self.ollama_model = None
        self.current_model_type = None
        
        logger.info("LLMHandler initialized")
    
    def configure_gemini(self, model_name: str = "gemini-2.0-flash") -> bool:
        """
        Configure Gemini model.
        
        Args:
            model_name: Name of the Gemini model to use
            
        Returns:
            bool: True if configuration successful
        """
        if not GEMINI_AVAILABLE:
            logger.error("Gemini not available. Install google-genai package.")
            return False
        
        api_key = os.getenv("GEMINI_API_KEY", "AIzaSyDxYogQnjYmTbvghsqSWCKtot5DpygC8hA")
        if not api_key:
            logger.error("GEMINI_API_KEY not found in environment variables")
            return False
        
        try:
            genai.configure(api_key=api_key)
            self.gemini_client = genai.GenerativeModel(model_name)

            self.current_model_type = "gemini"
            self.gemini_model_name = model_name
            logger.info(f"Gemini configured with model: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Error configuring Gemini: {e}")
            return False
    
    def configure_ollama(self, model_name: str, temperature: float = 0.3, max_tokens: int = 4000) -> bool:
        """
        Configure Ollama model with optimized settings for JSON generation.
        
        Args:
            model_name: Name of the Ollama model to use
            temperature: Temperature for generation (0.3 is optimal for structured output)
            max_tokens: Maximum tokens to generate (4000 for complex documents)
            
        Returns:
            bool: True if configuration successful
        """
        if not OLLAMA_AVAILABLE:
            logger.error("Ollama not available. Install langchain-ollama package.")
            return False
        
        try:
            self.ollama_model = ChatOllama(
                model=model_name,
                temperature=temperature,
                num_predict=max_tokens,
                # Additional parameters for better JSON generation
                format="json" if hasattr(ChatOllama, 'format') else None,
                # Ensure consistent output
                top_p=0.9,
                repeat_penalty=1.1
            )
            self.current_model_type = "ollama"
            self.ollama_model_name = model_name
            logger.info(f"Ollama configured with model: {model_name} (temp={temperature}, max_tokens={max_tokens})")
            return True
        except Exception as e:
            logger.error(f"Error configuring Ollama: {e}")
            return False
    
    def generate_response(self, prompt: str) -> str:
        """
        Generate response using the configured model.
        
        Args:
            prompt: Input prompt
            
        Returns:
            str: Generated response
        """
        if self.current_model_type == "gemini":
            return self._generate_gemini_response(prompt)
        elif self.current_model_type == "ollama":
            return self._generate_ollama_response(prompt)
        else:
            raise ValueError("No model configured. Please configure either Gemini or Ollama.")
    
    def _generate_gemini_response(self, prompt: str) -> str:
        try:
            response = self.gemini_client.generate_content(prompt)
            return response.text.strip() if response.text else "No response generated."
        except Exception as e:
            logger.error(f"Error generating Gemini response: {e}")
            return f"Error: {str(e)}"

    
    def _generate_ollama_response(self, prompt: str) -> str:
        """Generate response using Ollama with enhanced error handling."""
        try:
            logger.debug(f"Sending prompt to Ollama model {getattr(self, 'ollama_model_name', 'unknown')}")
            response = self.ollama_model.invoke(prompt)
            
            # Extract content from response
            content = ""
            if hasattr(response, "content"):
                content = response.content.strip()
            elif hasattr(response, "message"):
                content = response.message.strip()
            else:
                content = str(response).strip()
            
            if not content:
                logger.warning("Ollama returned empty response")
                return "Error: Empty response from Ollama model"
            
            logger.debug(f"Ollama response length: {len(content)} characters")
            return content
            
        except Exception as e:
            logger.error(f"Error generating Ollama response: {e}")
            # Provide more specific error information
            if "connection" in str(e).lower():
                return "Error: Cannot connect to Ollama. Please ensure Ollama is running."
            elif "model" in str(e).lower():
                return f"Error: Model issue - {str(e)}"
            else:
                return f"Error: {str(e)}"
    
    def get_available_ollama_models(self) -> List[str]:
        """
        Get list of available Ollama models.
        
        Returns:
            List of model names
        """
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                models = []
                for line in lines:
                    if line.strip():
                        model_name = line.split()[0]
                        models.append(model_name)
                return models
            else:
                logger.warning("Failed to get Ollama models list")
                return []
                
        except subprocess.TimeoutExpired:
            logger.warning("Timeout getting Ollama models")
            return []
        except FileNotFoundError:
            logger.warning("Ollama not found in system PATH")
            return []
        except Exception as e:
            logger.error(f"Error getting Ollama models: {e}")
            return []
    
    def is_model_configured(self) -> bool:
        """Check if any model is configured."""
        return self.current_model_type is not None
    
    def get_current_model_info(self) -> Dict:
        """Get information about the currently configured model."""
        if self.current_model_type == "gemini":
            return {
                "type": "gemini",
                "model": getattr(self, 'gemini_model_name', 'unknown'),
                "status": "online"
            }
        elif self.current_model_type == "ollama":
            return {
                "type": "ollama", 
                "model": "configured",
                "status": "offline"
            }
        else:
            return {
                "type": None,
                "model": None,
                "status": "not_configured"
            }