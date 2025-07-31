# import logging
# from typing import Dict, List
# from RAG1 import HybridSearchRAG, search_engine
# from main import extract_context_from_files, DOCUMENT_PATHS
# import time
# import json
# import re

# # Logging setup
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG)

# # Define regex patterns for greetings and closings
# GREETING_PATTERNS = [
#     r"^h+i+\b",           # hi, hii, hiiii, etc.
#     r"^h+e+y+\b",         # hey, heyy, heyyy, etc.
#     r"^h+e+l+o+\w*\b",    # hello, helloo, hellooo, hellow, hellooow, etc.
#     r"^good\s*morning\b",
#     r"^good\s*afternoon\b",
#     r"^good\s*evening\b"
# ]

# CLOSING_PATTERNS = [
#     r"thank\s*you\b", r"thanks\b", r"got\s*it\b", r"understood\b", r"bye+\b", r"goodbye\b",
#     r"see\s*you\b", r"i\s*got\s*it\b", r"ok\b", r"okay\b"
# ]


# # Define common greetings and closing remarks
# GREETING_PHRASES = ["hi","hiii","hiiii", "hello", "hey", "good morning", "good afternoon", "good evening"]
# CLOSING_PHRASES = ["thank you", "thanks", "got it", "understood", "bye", "goodbye", "see you", "i got it", "ok", "okay"]


# def is_greeting(query: str) -> bool:
#     q = query.lower().strip()
#     return any(q.startswith(greet) or q == greet for greet in GREETING_PHRASES)


# def is_closing(query: str) -> bool:
#     q = query.lower().strip()
#     return any(phrase in q for phrase in CLOSING_PHRASES)


# def is_vague_or_irrelevant(query: str, context: str) -> bool:
#     VAGUE_PHRASES = [
#         "tell me in details", "tell me about", "describe", "what is", "explain", "details about", "information about"
#     ]
#     q = query.lower()
#     # If context is not empty, do NOT treat as vague
#     if context and context.strip():
#         return False
#     # Otherwise, use the old logic
#     return any(phrase in q for phrase in VAGUE_PHRASES)


# def is_greeting_only(query: str) -> bool:
#     q = query.lower().strip()
#     q = re.sub(r'[^\w\s]', '', q)
#     # Only greeting if the whole query matches a greeting pattern (no extra words)
#     return any(re.fullmatch(pattern, q) for pattern in GREETING_PATTERNS)


# def is_closing_only(query: str) -> bool:
#     q = query.lower().strip()
#     q = re.sub(r'[^\w\s]', '', q)
#     return any(re.search(pattern, q) for pattern in CLOSING_PATTERNS)


# def query_starts_with_greeting(query: str) -> bool:
#     q = query.lower().strip()
#     q = re.sub(r'[^\w\s]', '', q)
#     return any(re.match(pattern, q) for pattern in GREETING_PATTERNS)


# def is_greeting_with_filler_only(query: str) -> bool:
#     """
#     Returns True if the query starts with a greeting and is followed only by 1-2 non-question, non-SFA words.
#     This makes the greeting handling dynamic and robust to new filler words.
#     """
#     q = query.lower().strip()
#     q = re.sub(r'[^\w\s]', '', q)
#     words = q.split()
#     # Only check if more than one word
#     if len(words) == 1:
#         return False
#     # If first word is a greeting and the rest are not question/SFA words
#     if any(re.fullmatch(pattern, words[0]) for pattern in GREETING_PATTERNS):
#         for w in words[1:]:
#             if w in {"what", "how", "why", "when", "where", "who", "sfa", "application", "software", "sales", "force", "automation"}:
#                 return False
#         # If only 1 or 2 such words, treat as greeting+filler
#         return len(words) <= 3
#     return False


# def process_query(
#         query: str,
#         min_relevance_score: float = 0.5,
#         model_choice: str = "gemini",
#         gemini_model=None,
#         qwen_model=None
# ) -> Dict:
#     logger.debug(f"Processing query: {query} with model: {model_choice}")

#     # 1. Handle greetings ONLY if the query is just a greeting or greeting+filler
#     if is_greeting_only(query) or is_greeting_with_filler_only(query):
#         answer = "Hello! How can I assist you about SFA?"
#         return {
#             "answer": answer,
#             "relevance_scores": [],
#             "matched_documents": []
#         }

#     # 2. Handle closings ONLY if the query is just a closing
#     if is_closing_only(query):
#         answer = "Goodbye!"
#         return {
#             "answer": answer,
#             "relevance_scores": [],
#             "matched_documents": []
#         }

#     # Perform the search using HybridSearchRAG
#     try:
#         results = search_engine.query_search(query, min_relevance_score=min_relevance_score)
#         if len(results) == 0:
#             return {
#                 "answer": "I cannot answer this question. Please ask a specific question about the SFA application."
#             }
#         logger.debug(f"\nSearch results: \n\n{results}")
#         for idx, result in enumerate(results):
#             logger.debug(
#                 f"\nResult {idx + 1}: \n\n[Score={result.get('relevance_score')}], Text={result.get('text')[:]}.\n")
#     except Exception as e:
#         logger.error(f"Error performing search: {e}")
#         return {
#             "answer": "Sorry, something went wrong while searching.",
#             "relevance_scores": [],
#             "matched_documents": []
#         }

#     combined_context = "\n".join([result['text'] for result in results])

#     # 3. Strict enforcement for vague/irrelevant queries
#     if is_vague_or_irrelevant(query, combined_context):
#         answer = "Please ask a specific question about the SFA application."
#         return {
#             "answer": answer,
#             "relevance_scores": [result['relevance_score'] for result in results],
#             "matched_documents": [result['text'] for result in results]
#         }

#     # Prepare prompt for the model with updated instructions
#     prompt = (
#         "You are a professional multi-language supported customer support assistant for SFA (Sales Force Automation).\n"
#         "If the user refers to 'this software', 'this application', or similar terms, interpret this as referring to the SFA Application.\n"
#         "Answer using ONLY the information in the Document Context provided below.\n"
#         "Do NOT use external knowledge or make assumptions.\n"
#         "Keep all responses concise, professional, and directly relevant to the user's question.\n"
#         "If the user includes a greeting (e.g., 'Hello!', 'Hi'), start your response with a brief greeting (e.g., 'Hello!').\n"
#         "If the question is vague, overly broad (e.g., 'tell me in details'), not clearly related to SFA, or consists of gibberish (excluding greetings), reply ONLY with:\n"
#         "'I cannot answer this question. Please ask a specific question about the SFA application.'\n"
#         "If the answer cannot be found in the Document Context or the context does not directly address the question, reply ONLY with:\n"
#         "'I cannot answer this question. Please ask a specific question about the SFA application.'\n"
#         "Only answer specific, clear SFA-related questions that can be directly addressed using the provided context.\n"
#         "Under NO circumstances should you include the user question, document context, or any debug information in your response.\n"
#         "Do NOT mention or repeat the user question or document context in the response.\n\n"
#         f"Document Context:\n{combined_context}\n\n"
#         f"User Question: {query}"
#     )
#     print(f"\033[1;31mCombined context:\n\n{combined_context}\033[0m")
#     logger.debug(f"Prompt sent to model:\n{prompt}")

#     # Process based on model choice
#     if model_choice == "gemini":
#         # Query Gemini API
#         try:
#             response = gemini_model.generate_content(prompt)
#             answer = response.text.strip()
#         except Exception as e:
#             logger.error(f"Error generating content from Gemini: {e}")
#             answer = "Sorry, something went wrong while generating the answer."
#         logger.debug(f"\nGemini response: {answer}")

#     elif model_choice == "llama":
#         # Query Qwen2.5 API
#         try:
#             qwen_response = qwen_model.invoke(prompt)
#             if hasattr(qwen_response, "content"):
#                 answer = qwen_response.content.strip()
#             elif hasattr(qwen_response, "message"):
#                 answer = qwen_response.message.strip()
#             else:
#                 answer = str(qwen_response).strip()
#         except Exception as e:
#             logger.error(f"Error generating content from Qwen2.5: {e}")
#             answer = "Sorry, something went wrong while generating the answer."
#         logger.debug(f"\nQwen2.5 response:\n {answer}")

#     else:
#         logger.error(f"Invalid model_choice: {model_choice}")
#         answer = "Invalid model choice. Please use 'gemini' or 'qwen'."

#     # Add greeting if the query starts with a greeting and is not greeting-only
#     if query_starts_with_greeting(query) and not is_greeting_only(query):
#         # Only prepend if the answer doesn't already start with a greeting
#         if not re.match(r"^\s*hello[!.,\s]*", answer, re.IGNORECASE):
#             answer = f"Hello! {answer}"

#     # Prepare response
#     return {
#         "answer": answer,
#         "relevance_scores": [result['relevance_score'] for result in results],
#         "matched_documents": [result['text'] for result in results]
#     }


# import logging
# from typing import Dict, List
# import re
# import json

# # Logging setup
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG)

# # Define regex patterns for greetings and closings
# GREETING_PATTERNS = [
#     r"^h+i+\b", r"^h+e+y+\b", r"^h+e+l+o+\w*\b", r"^good\s*morning\b", r"^good\s*afternoon\b", r"^good\s*evening\b",
#     r"^greetings\b", r"^welcome\b", r"^howdy\b", r"^yo+\b", r"^sup\b", r"^what'?s?\s*up\b", r"^salutations\b",
#     r"^dear\b", r"^to\s*whom\s*it\s*may\s*concern\b", r"^hello\s*there\b", r"^hi\s*there\b", r"^hey\s*there\b",
#     r"^good\s*day\b", r"^peace\b", r"^shalom\b", r"^namaste\b", r"^bonjour\b", r"^hola\b", r"^ciao\b", r"^hallo\b",
#     r"^aloha\b", r"^yo\b", r"^hey\s*everyone\b", r"^hi\s*everyone\b", r"^hey\s*all\b", r"^hi\s*all\b",
#     r"^hey\s*guys\b", r"^hi\s*guys\b", r"^hey\s*team\b", r"^hi\s*team\b",
#     r"^wassup\b", r"^hiya\b", r"^hey\s*there\b", r"^hi\s*there\b", r"^hey\s*folks\b", r"^hi\s*folks\b",
#     r"^hey\s*friends\b", r"^hi\s*friends\b", r"^hey\s*buddy\b", r"^hi\s*buddy\b", r"^hey\s*pal\b", r"^hi\s*pal\b",
#     r"^hey\s*everyone\b", r"^hi\s*everyone\b", r"^hey\s*all\b", r"^hi\s*all\b", r"^hey\s*people\b", r"^hi\s*people\b",
#     r"^hey\s*there\b", r"^hi\s*there\b", r"^yo+\b", r"^sup\b", r"^hey\s*yall\b", r"^hi\s*yall\b",
#     r"^heya+\b", r"^heyaa+\b", r"^heyaaaa+\b", r"^heya\s*there\b", r"^heya\s*everyone\b", r"^heya\s*all\b",
#     r"^heya\s*guys\b", r"^heya\s*team\b", r"^heya\s*folks\b", r"^heya\s*friends\b", r"^heya\s*buddy\b",
#     r"^heya\s*pal\b", r"^heya\s*people\b", r"^heya\s*yall\b",
#     r"^hlw+\b", r"^hlww+\b", r"^hlwww+\b", r"^hlw\s*there\b", r"^hlw\s*everyone\b", r"^hlw\s*all\b",
#     r"^hlw\s*guys\b", r"^hlw\s*team\b", r"^hlw\s*folks\b", r"^hlw\s*friends\b", r"^hlw\s*buddy\b",
#     r"^hlw\s*pal\b", r"^hlw\s*people\b", r"^hlw\s*yall\b"
# ]

# CLOSING_PATTERNS = [
#     r"\bthank\s*u\b", r"\bthanks\b", r"\bthank\s*you\b", r"\bthx\b", r"\bty\b", r"\bthnks\b",  # <-- added thnks
#     r"\bappreciate\s*it\b",
#     r"\bgot\s*this\b", r"\bi\s*got\s*this\b", r"\bgot\s*it\b", r"\bi\s*got\s*it\b", r"\bunderstood\b",
#     r"\bbye+\b", r"\bgoodbye\b", r"\bsee\s*you\b", r"\bsee\s*ya\b", r"\bsee\s*ya\s*later\b",
#     r"\bcatch\s*you\s*later\b", r"\blater\b", r"\btake\s*care\b", r"\bhave\s*a\s*nice\s*day\b",
#     r"\bhave\s*a\s*good\s*day\b", r"\bhave\s*a\s*great\s*day\b", r"\bciao\b", r"\badios\b",
#     r"\bok\b", r"\bokay\b", r"\bpeace\b", r"\bcheers\b", r"\bsee\s*you\s*soon\b", r"\bsee\s*you\s*next\s*time\b",
#     r"\bfarewell\b", r"\bso\s*long\b", r"\bsee\s*ya\b", r"\bbye\s*bye\b", r"\bbye\s*for\s*now\b",
#     r"\bcatch\s*ya\s*later\b", r"\buntil\s*next\s*time\b", r"\bim\s*out\b", r"\bi\s*am\s*out\b",
#     r"\bpeace\s*out\b", r"\btoodles\b", r"\bsee\s*you\s*around\b", r"\bsee\s*you\s*again\b",
#     r"\bsee\s*you\s*later\s*alligator\b", r"\bafter\s*while\s*crocodile\b", r"\bim\s*off\b", r"\bi\s*am\s*off\b",
#     r"\bbye\s*now\b", r"\bsee\s*ya\s*soon\b", r"\bsee\s*ya\s*next\s*time\b",
#     r"\bdone\b", r"\bdone\s*for\s*the\s*day\b", r"\bdone\s*for\s*now\b", r"\bfinished\b", r"\bfinished\s*for\s*the\s*day\b"
# ]
# GREETING_PHRASES = [
#     "hi", "hii", "hiii", "hiiii", "hello", "helo", "helloo", "hellooo", "hey", "heyy", "heyyy", "hlw",
#     "heya", "heyaa", "heyaaa", "heyaaaa", "heyas", "heya there", "heya everyone", "heya all", "heya guys", "heya team", "heya folks", "heya friends", "heya buddy", "heya pal", "heya people", "heya yall",
#     "good morning", "good afternoon", "good evening", "greetings", "welcome", "howdy", "yo", "yo!", "sup",
#     "whats up", "what's up", "salutations", "dear", "hi there", "hello there", "hey there", "good day",
#     "peace", "shalom", "namaste", "bonjour", "hola", "ciao", "hallo", "aloha", "hiya", "wassup",
#     "hey everyone", "hi everyone", "hey all", "hi all", "hey guys", "hi guys", "hey team", "hi team",
#     "hey folks", "hi folks", "hey friends", "hi friends", "hey buddy", "hi buddy", "hey pal", "hi pal",
#     "hey people", "hi people", "hey yall", "hi yall"
# ]
# CLOSING_PHRASES = [
#     "thank you", "thanks", "thnks", "got it", "understood", "bye", "goodbye", "see you", "i got it", "ok", "okay", "thanku", "thnx", "appreciate it", "got this", "got it", "bye bye", "see ya", "see ya later",
#     "catch you later", "later", "take care", "have a nice day"
# ]
# def is_greeting(query: str) -> bool:
#     q = query.lower().strip()
#     return any(q.startswith(greet) or q == greet for greet in GREETING_PHRASES)

# def is_closing(query: str) -> bool:
#     q = query.lower().strip()
#     return any(phrase in q for phrase in CLOSING_PHRASES)

# def is_vague_or_irrelevant(query: str, context: str) -> bool:
#     VAGUE_PHRASES = [
#         "tell me in details", "tell me about", "describe", "what is", "explain", "details about", "information about"
#     ]
#     q = query.lower()
#     if context and context.strip():
#         return False
#     return any(phrase in q for phrase in VAGUE_PHRASES)

# def is_greeting_only(query: str) -> bool:
#     q = query.lower().strip()
#     q = re.sub(r'[^\w\s]', '', q)
#     return any(re.fullmatch(pattern, q) for pattern in GREETING_PATTERNS)

# def is_closing_only(query: str) -> bool:
#     q = query.lower().strip()
#     q = re.sub(r'[^\w\s]', '', q)
#     words = q.split()
#     # If all words are closing/filler, treat as closing
#     closing_words = {"thanku","thnx", "thanks", "thank", "you", "got", "this", "it", "understood", "bye", "goodbye", "see", "ok", "okay"}
#     if all(w in closing_words for w in words):
#         return True
#     # Or match with regex
#     return any(re.search(pattern, q) for pattern in CLOSING_PATTERNS)

# def query_starts_with_greeting(query: str) -> bool:
#     q = query.lower().strip()
#     q = re.sub(r'[^\w\s]', '', q)
#     for pattern in GREETING_PATTERNS:
#         if re.match(pattern, q):
#             logger.debug(f"Greeting pattern matched: {pattern} for query: {q}")
#             return True
#     return False

# def is_greeting_with_filler_only(query: str) -> bool:
#     q = query.lower().strip()
#     q = re.sub(r'[^\w\s]', '', q)
#     words = q.split()
#     if len(words) == 1:
#         return False
#     if any(re.fullmatch(pattern, words[0]) for pattern in GREETING_PATTERNS):
#         for w in words[1:]:
#             if w in {"what", "how", "why", "when", "where", "who", "sfa", "application", "software", "sales", "force", "automation"}:
#                 return False
#         return len(words) <= 3
#     return False

# def process_query(
#         query: str,
#         min_relevance_score: float = 0.5,
#         model_choice: str = "gemini",
#         gemini_model=None,
#         qwen_model=None,
#         search_engine=None
# ) -> Dict:
#     logger.debug(f"Processing query: {query} with model: {model_choice}")

#     # Handle greetings only if the query is just a greeting or greeting+filler
#     if is_greeting_only(query) or is_greeting_with_filler_only(query):
#         answer = "Hello! How can I assist you about SFA?"
#         return {
#             "answer": answer,
#             "relevance_scores": [],
#             "matched_documents": []
#         }

#     # Handle closings only if the query is just a closing
#     if is_closing_only(query):
#         answer = "Thank you for reaching out. If you have any more questions about the SFA application in the future, feel free to ask. Have a great day!"
#         return {
#             "answer": answer,
#             "relevance_scores": [],
#             "matched_documents": []
#         }

#     # Perform the search using HybridSearchRAG
#     try:
#         results = search_engine.query_search(query, min_relevance_score=min_relevance_score)
#         if len(results) == 0:
#             return {
#                 "answer": "I cannot answer this question. Please ask a specific question about the SFA application."
#             }
#         logger.debug(f"\nSearch results: \n\n{results}")
#         for idx, result in enumerate(results):
#             logger.debug(
#                 f"\nResult {idx + 1}: \n\n[Score={result.get('relevance_score')}], Text={result.get('text')[:]}.\n")
#     except Exception as e:
#         logger.error(f"Error performing search: {e}")
#         return {
#             "answer": "Sorry, something went wrong while searching.",
#             "relevance_scores": [],
#             "matched_documents": []
#         }

#     combined_context = "\n".join([result['text'] for result in results])

#     # Strict enforcement for vague/irrelevant queries
#     if is_vague_or_irrelevant(query, combined_context):
#         answer = "Please ask a specific question about the SFA application."
#         return {
#             "answer": answer,
#             "relevance_scores": [result['relevance_score'] for result in results],
#             "matched_documents": [result['text'] for result in results]
#         }

#     # Prepare prompt for the model
#     prompt = (
#         "You are a professional multi-language supported customer support assistant for SFA (Sales Force Automation).\n"
#         "If the user refers to 'this software', 'this application', or similar terms, interpret this as referring to the SFA Application.\n"
#         "Answer using ONLY the information in the Document Context provided below.\n"
#         "Do NOT use external knowledge or make assumptions.\n"
#         "Keep all responses concise, professional, and directly relevant to the user's question.\n"
#         # "If the user includes a greeting (e.g., 'Hello!', 'Hi'), start your response with a brief greeting (e.g., 'Hello!').\n"
#         "If the question is vague, overly broad (e.g., 'tell me in details'), not clearly related to SFA, or consists of gibberish (excluding greetings), reply ONLY with:\n"
#         "'I cannot answer this question. Please ask a specific question about the SFA application.'\n"
#         "If the answer cannot be found in the Document Context or the context does not directly address the question, reply ONLY with:\n"
#         "'I cannot answer this question. Please ask a specific question about the SFA application.'\n"
#         "Only answer specific, clear SFA-related questions that can be directly addressed using the provided context.\n"
#         "Under NO circumstances should you include the user question, document context, or any debug information in your response.\n"
#         "Do NOT mention or repeat the user question or document context in the response.\n\n"
#         f"Document Context:\n{combined_context}\n\n"
#         f"User Question: {query}"
#     )
#     print(f"\033[1;31mCombined context:\n\n{combined_context}\033[0m")
#     logger.debug(f"Prompt sent to model:\n{prompt}")

#     # Process based on model choice
#     if model_choice == "gemini":
#         try:
#             response = gemini_model.generate_content(prompt)
#             answer = response.text.strip()
#         except Exception as e:
#             logger.error(f"Error generating content from Gemini: {e}")
#             answer = "Sorry, something went wrong while generating the answer."
#         logger.debug(f"\nGemini response: {answer}")
#     elif model_choice == "llama":
#         try:
#             qwen_response = qwen_model.invoke(prompt)
#             if hasattr(qwen_response, "content"):
#                 answer = qwen_response.content.strip()
#             elif hasattr(qwen_response, "message"):
#                 answer = qwen_response.message.strip()
#             else:
#                 answer = str(qwen_response).strip()
#         except Exception as e:
#             logger.error(f"Error generating content from Qwen2.5: {e}")
#             answer = "Sorry, something went wrong while generating the answer."
#         logger.debug(f"\nQwen2.5 response:\n {answer}")
#     else:
#         logger.error(f"Invalid model_choice: {model_choice}")
#         answer = "Invalid model choice. Please use 'gemini' or 'llama'."

#     # Add greeting if the query starts with a greeting and is not greeting-only
#     if query_starts_with_greeting(query) and not is_greeting_only(query):
#         if not re.match(r"^\s*hello[!.,\s]*", answer, re.IGNORECASE):
#             answer = f"Hello! {answer}"

#     # Prepare response
#     return {
#         "answer": answer,
#         "relevance_scores": [result['relevance_score'] for result in results],
#         "matched_documents": [result['text'] for result in results]
#     }

import logging
from typing import Dict, List
import re
import json

# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# Define regex patterns for greetings and closings (unchanged)
GREETING_PATTERNS = [
    r"^h+i+\b", r"^h+e+y+\b", r"^h+e+l+o+\w*\b", r"^good\s*morning\b", r"^good\s*afternoon\b", r"^good\s*evening\b",
    r"^greetings\b", r"^welcome\b", r"^howdy\b", r"^yo+\b", r"^sup\b", r"^what'?s?\s*up\b", r"^salutations\b",
    r"^dear\b", r"^to\s*whom\s*it\s*may\s*concern\b", r"^hello\s*there\b", r"^hi\s*there\b", r"^hey\s*there\b",
    r"^good\s*day\b", r"^peace\b", r"^shalom\b", r"^namaste\b", r"^bonjour\b", r"^hola\b", r"^ciao\b", r"^hallo\b",
    r"^aloha\b", r"^yo\b", r"^hey\s*everyone\b", r"^hi\s*everyone\b", r"^hey\s*all\b", r"^hi\s*all\b",
    r"^hey\s*guys\b", r"^hi\s*guys\b", r"^hey\s*team\b", r"^hi\s*team\b",
    r"^wassup\b", r"^hiya\b", r"^hey\s*there\b", r"^hi\s*there\b", r"^hey\s*folks\b", r"^hi\s*folks\b",
    r"^hey\s*friends\b", r"^hi\s*friends\b", r"^hey\s*buddy\b", r"^hi\s*buddy\b", r"^hey\s*pal\b", r"^hi\s*pal\b",
    r"^hey\s*everyone\b", r"^hi\s*everyone\b", r"^hey\s*all\b", r"^hi\s*all\b", r"^hey\s*people\b", r"^hi\s*people\b",
    r"^hey\s*there\b", r"^hi\s*there\b", r"^yo+\b", r"^sup\b", r"^hey\s*yall\b", r"^hi\s*yall\b",
    r"^heya+\b", r"^heyaa+\b", r"^heyaaaa+\b", r"^heya\s*there\b", r"^heya\s*everyone\b", r"^heya\s*all\b",
    r"^heya\s*guys\b", r"^heya\s*team\b", r"^heya\s*folks\b", r"^heya\s*friends\b", r"^heya\s*buddy\b",
    r"^heya\s*pal\b", r"^heya\s*people\b", r"^heya\s*yall\b",
    r"^hlw+\b", r"^hlww+\b", r"^hlwww+\b", r"^hlw\s*there\b", r"^hlw\s*everyone\b", r"^hlw\s*all\b",
    r"^hlw\s*guys\b", r"^hlw\s*team\b", r"^hlw\s*folks\b", r"^hlw\s*friends\b", r"^hlw\s*buddy\b",
    r"^hlw\s*pal\b", r"^hlw\s*people\b", r"^hlw\s*yall\b"
]

CLOSING_PATTERNS = [
    r"\bthank\s*u\b", r"\bthanks\b", r"\bthank\s*you\b", r"\bthx\b", r"\bty\b", r"\bthnks\b",
    r"\bappreciate\s*it\b",
    r"\bgot\s*this\b", r"\bi\s*got\s*this\b", r"\bgot\s*it\b", r"\bi\s*got\s*it\b", r"\bunderstood\b",
    r"\bbye+\b", r"\bgoodbye\b", r"\bsee\s*you\b", r"\bsee\s*ya\b", r"\bsee\s*ya\s*later\b",
    r"\bcatch\s*you\s*later\b", r"\blater\b", r"\btake\s*care\b", r"\bhave\s*a\s*nice\s*day\b",
    r"\bhave\s*a\s*good\s*day\b", r"\bhave\s*a\s*great\s*day\b", r"\bciao\b", r"\badios\b",
    r"\bok\b", r"\bokay\b", r"\bpeace\b", r"\bcheers\b", r"\bsee\s*you\s*soon\b", r"\bsee\s*you\s*next\s*time\b",
    r"\bfarewell\b", r"\bso\s*long\b", r"\bsee\s*ya\b", r"\bbye\s*bye\b", r"\bbye\s*for\s*now\b",
    r"\bcatch\s*ya\s*later\b", r"\buntil\s*next\s*time\b", r"\bim\s*out\b", r"\bi\s*am\s*out\b",
    r"\bpeace\s*out\b", r"\btoodles\b", r"\bsee\s*you\s*around\b", r"\bsee\s*you\s*again\b",
    r"\bsee\s*you\s*later\s*alligator\b", r"\bafter\s*while\s*crocodile\b", r"\bim\s*off\b", r"\bi\s*am\s*off\b",
    r"\bbye\s*now\b", r"\bsee\s*ya\s*soon\b", r"\bsee\s*ya\s*next\s*time\b",
    r"\bdone\b", r"\bdone\s*for\s*the\s*day\b", r"\bdone\s*for\s*now\b", r"\bfinished\b", r"\bfinished\s*for\s*the\s*day\b"
]

GREETING_PHRASES = [
    "hi", "hii", "hiii", "hiiii", "hello", "helo", "helloo", "hellooo", "hey", "heyy", "heyyy", "hlw",
    "heya", "heyaa", "heyaaa", "heyaaaa", "heyas", "heya there", "heya everyone", "heya all", "heya guys", "heya team", "heya folks", "heya friends", "heya buddy", "heya pal", "heya people", "heya yall",
    "good morning", "good afternoon", "good evening", "greetings", "welcome", "howdy", "yo", "yo!", "sup",
    "whats up", "what's up", "salutations", "dear", "hi there", "hello there", "hey there", "good day",
    "peace", "shalom", "namaste", "bonjour", "hola", "ciao", "hallo", "aloha", "hiya", "wassup",
    "hey everyone", "hi everyone", "hey all", "hi all", "hey guys", "hi guys", "hey team", "hi team",
    "hey folks", "hi folks", "hey friends", "hi friends", "hey buddy", "hi buddy", "hey pal", "hi pal",
    "hey people", "hi people", "hey yall", "hi yall"
]

CLOSING_PHRASES = [
    "thank you", "thanks", "thnks", "got it", "understood", "bye", "goodbye", "see you", "i got it", "ok", "okay", "thanku", "thnx", "appreciate it", "got this", "got it", "bye bye", "see ya", "see ya later",
    "catch you later", "later", "take care", "have a nice day"
]

def is_greeting(query: str) -> bool:
    q = query.lower().strip()
    return any(q.startswith(greet) or q == greet for greet in GREETING_PHRASES)

def is_closing(query: str) -> bool:
    q = query.lower().strip()
    return any(phrase in q for phrase in CLOSING_PHRASES)

def is_vague_or_irrelevant(query: str, context: str) -> bool:
    VAGUE_PHRASES = [
        "tell me in details", "tell me about", "describe", "what is", "explain", "details about", "information about"
    ]
    q = query.lower()
    if context and context.strip():
        return False
    return any(phrase in q for phrase in VAGUE_PHRASES)

def is_greeting_only(query: str) -> bool:
    q = query.lower().strip()
    q = re.sub(r'[^\w\s]', '', q)
    return any(re.fullmatch(pattern, q) for pattern in GREETING_PATTERNS)

def is_closing_only(query: str) -> bool:
    q = query.lower().strip()
    q = re.sub(r'[^\w\s]', '', q)
    words = q.split()
    closing_words = {"thanku", "thnx", "thanks", "thank", "you", "got", "this", "it", "understood", "bye", "goodbye", "see", "ok", "okay"}
    if all(w in closing_words for w in words):
        return True
    return any(re.search(pattern, q) for pattern in CLOSING_PATTERNS)

def query_starts_with_greeting(query: str) -> bool:
    q = query.lower().strip()
    q = re.sub(r'[^\w\s]', '', q)
    for pattern in GREETING_PATTERNS:
        if re.match(pattern, q):
            logger.debug(f"Greeting pattern matched: {pattern} for query: {q}")
            return True
    return False

def is_greeting_with_filler_only(query: str) -> bool:
    q = query.lower().strip()
    q = re.sub(r'[^\w\s]', '', q)
    words = q.split()
    if len(words) == 1:
        return False
    if any(re.fullmatch(pattern, words[0]) for pattern in GREETING_PATTERNS):
        for w in words[1:]:
            if w in {"what", "how", "why", "when", "where", "who", "sfa", "application", "software", "sales", "force", "automation"}:
                return False
        return len(words) <= 3
    return False

def process_query(
        query: str,
        min_relevance_score: float = 0.5,
        model_choice: str = "gemini",
        gemini_model=None,
        qwen_model=None,
        search_engine=None,
        topic: str = "Unknown Topic"
) -> Dict:
    logger.debug(f"Processing query: {query} with model: {model_choice} and topic: {topic}")

    # Handle greetings only if the query is just a greeting or greeting+filler
    if is_greeting_only(query) or is_greeting_with_filler_only(query):
        answer = f"Hello! How can I assist you about {topic}?"
        return {
            "answer": answer,
            "relevance_scores": [],
            "matched_documents": []
        }

    # Handle closings only if the query is just a closing
    if is_closing_only(query):
        answer = f"Thank you for reaching out. If you have any more questions about {topic} in the future, feel free to ask. Have a great day!"
        return {
            "answer": answer,
            "relevance_scores": [],
            "matched_documents": []
        }

    # Perform the search using HybridSearchRAG
    try:
        results = search_engine.query_search(query, min_relevance_score=min_relevance_score)
        if len(results) == 0:
            return {
                "answer": f"I cannot answer this question. Please ask a specific question about {topic}."
            }
        logger.debug(f"\nSearch results: \n\n{results}")
        for idx, result in enumerate(results):
            logger.debug(
                f"\nResult {idx + 1}: \n\n[Score={result.get('relevance_score')}], Text={result.get('text')[:]}.\n")
    except Exception as e:
        logger.error(f"Error performing search: {e}")
        return {
            "answer": "Sorry, something went wrong while searching.",
            "relevance_scores": [],
            "matched_documents": []
        }

    combined_context = "\n".join([result['text'] for result in results])

    # Strict enforcement for vague/irrelevant queries
    if is_vague_or_irrelevant(query, combined_context):
        answer = f"Please ask a specific question about {topic}."
        return {
            "answer": answer,
            "relevance_scores": [result['relevance_score'] for result in results],
            "matched_documents": [result['text'] for result in results]
        }

    # Prepare prompt for the model
    prompt = (
        f"You are a professional multi-language supported customer support assistant for {topic}.\n"
        "If the user refers to 'this software', 'this application', or similar terms, interpret this as referring to the application or system related to {topic}.\n"
        "Answer using ONLY the information in the Document Context provided below.\n"
        "Do NOT use external knowledge or make assumptions.\n"
        "Keep all responses concise, professional, and directly relevant to the user's question.\n"
        "If the question is vague, overly broad (e.g., 'tell me in details'), not clearly related to {topic}, or consists of gibberish (excluding greetings), reply ONLY with:\n"
        f"'I cannot answer this question. Please ask a specific question about {topic}.'\n"
        f"If the answer cannot be found in the Document Context or the context does not directly address the question, reply ONLY with:\n"
        f"'I cannot answer this question. Please ask a specific question about {topic}.'\n"
        f"Only answer specific, clear {topic}-related questions that can be directly addressed using the provided context.\n"
        "Under NO circumstances should you include the user question, document context, or any debug information in your response.\n"
        "Do NOT mention or repeat the user question or document context in the response.\n\n"
        f"Document Context:\n{combined_context}\n\n"
        f"User Question: {query}"
    )
    print(f"\033[1;31mCombined context:\n\n{combined_context}\033[0m")
    logger.debug(f"Prompt sent to model:\n{prompt}")

    # Process based on model choice
    if model_choice == "gemini":
        try:
            response = gemini_model.generate_content(prompt)
            answer = response.text.strip()
        except Exception as e:
            logger.error(f"Error generating content from Gemini: {e}")
            answer = "Sorry, something went wrong while generating the answer."
        logger.debug(f"\nGemini response: {answer}")
    elif model_choice == "llama":
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
        answer = "Invalid model choice. Please use 'gemini' or 'llama'."

    # Add greeting if the query starts with a greeting and is not greeting-only
    if query_starts_with_greeting(query) and not is_greeting_only(query):
        if not re.match(r"^\s*hello[!.,\s]*", answer, re.IGNORECASE):
            answer = f"Hello! {answer}"

    # Prepare response
    return {
        "answer": answer,
        "relevance_scores": [result['relevance_score'] for result in results],
        "matched_documents": [result['text'] for result in results]
    }