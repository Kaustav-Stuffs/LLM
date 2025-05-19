import logging
from typing import Dict, List
from RAG1 import HybridSearchRAG, search_engine
from main import extract_context_from_files, DOCUMENT_PATHS
import time
import json
# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
def process_query(
    query: str,
    min_relevance_score: float = 0.5,
    model_choice: str = "gemini",
    gemini_model=None,
    qwen_model=None
) -> Dict:
    """
    Process a query using HybridSearchRAG and either Gemini Flash API or Qwen2.5.
    
    Args:
        query: The search query from the customer support endpoint
        min_relevance_score: Minimum relevance score for search results
        model_choice: "gemini" or "qwen" to select the model
        gemini_model: Gemini model instance
        qwen_model: Qwen2.5 model instance
    
    Returns:
        Dictionary containing the answer and metadata
    """
    logger.debug(f"Processing query: {query} with model: {model_choice}")
    
    # Perform the search using HybridSearchRAG
    try:
        results = search_engine.query_search(query, min_relevance_score=min_relevance_score)
        logger.debug(f"\nSearch results: \n\n{results}")
        for idx, result in enumerate(results):
            logger.debug(f"\nResult {idx+1}: \n\n[Score={result.get('relevance_score')}], Text={result.get('text')[:]}.\n")
    except Exception as e:
        logger.error(f"Error performing search: {e}")
        return {
            "answer": "Sorry, something went wrong while searching for relevant information.",
            "relevance_scores": [],
            "matched_documents": []
        }

    # Extract context from documents (same as main.py)
    context = extract_context_from_files(DOCUMENT_PATHS)
    
    # Use only the search results as context for the model
    combined_context = "\n".join([result['text'] for result in results])
    
    # Prepare prompt for the model
    prompt = (
        "You are a professional and polite customer support assistant.\n"
        "Answer the user's question strictly using ONLY the information provided in the Document Context below.\n"
        "Do NOT use any external knowledge or information that is not present in the Document Context.\n"
        "If the provided context is relevant or if you will find the answer inside the context like if there is no full form availale in context than not provide any.\n"
        "If the answer cannot be found in the Document Context, reply with:\n"
        "\"I am not sure about that question. You can ask me another question.\"\n"
        "If the question is unclear or unrelated to the Document Context, reply with:\n"
        "\"I couldn't understand your question. Could you please rephrase it more clearly?\"\n\n"
        f"Document Context:\n{combined_context}\n\n"
        f"User Question: {query}"
    )
    # prompt = (
    #     f"Use only the following context to answer the question.\n"
    #     f"If the question is not answerable from this context, say 'I don't have enough information to answer that.'\n"
    #     f"\nContext:\n{combined_context}\n\nQuestion: {query}"
    # )
    # logger.debug(f"Prompt sent to model:\n{prompt}")
    
    # Process based on model choice
    if model_choice == "gemini":
        # Query Gemini API
        try:
            response = gemini_model.generate_content(prompt)
            answer = response.text.strip()
        except Exception as e:
            logger.error(f"Error generating content from Gemini: {e}")
            answer = "Sorry, something went wrong while generating the answer."
        logger.debug(f"\nGemini response: {answer}")
    
    elif model_choice == "llama":
        # Query Qwen2.5 API
        try:
            qwen_response = qwen_model.invoke(prompt)
            if hasattr(qwen_response, "content"):
                answer = qwen_response.content.strip()
            elif hasattr(qwen_response, "message"):
                answer = qwen_response.message.strip()
            else:
                answer = str(qwen_response).strip()
        except Exception as e:
            logger.error(f"Error generating content from Qwen2.5: {e}")
            answer = "Sorry, something went wrong while generating the answer."
        logger.debug(f"\nQwen2.5 response:\n {answer}")
    
    else:
        logger.error(f"Invalid model_choice: {model_choice}")
        answer = "Invalid model choice. Please use 'gemini' or 'qwen'."
    
    # Prepare response
    return {
        "answer": answer,
        "relevance_scores": [result['relevance_score'] for result in results],
        "matched_documents": [result['text'] for result in results]
    }