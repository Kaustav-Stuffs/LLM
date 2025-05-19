import json
import csv
import fitz  # PyMuPDF
import os
from typing import List, Dict
import logging
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text by removing stop words and punctuation.
    """
    stop_words = set(stopwords.words('english')).union({'question', 'answer'})
    words = word_tokenize(text.lower())
    # Remove punctuation and stop words
    words = [word for word in words if word not in string.punctuation and word not in stop_words]
    # Count word frequency
    word_freq = {}
    for word in words:
        if word.isalnum():  # Only include alphanumeric words
            word_freq[word] = word_freq.get(word, 0) + 1
    # Sort by frequency and limit to max_keywords
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_words[:max_keywords]]

def extract_text_from_file(file_path: str) -> List[str]:
    """
    Extract text from a file and split into chunks based on file type.
    Returns a list of text chunks.
    """
    logger.info(f"Processing file: {file_path}")
    try:
        if file_path.endswith(".txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                # Split by double newlines or paragraphs
                return [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]

        elif file_path.endswith(".json"):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return [json.dumps(item) for item in data]
                elif isinstance(data, dict):
                    return [json.dumps(data)]
                else:
                    return [str(data)]

        elif file_path.endswith(".csv"):
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                # Join each row into a single string
                return [", ".join(row).strip() for row in reader if row]

        elif file_path.endswith(".pdf"):
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            # Split by double newlines or paragraphs
            return [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]

        else:
            logger.error(f"Unsupported file format: {file_path}")
            return []

    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        return []

def convert_to_doc1_format(file_paths: List[str], output_path: str, source: str = "document", category: str = "General") -> None:
    """
    Convert documents to the doc1.txt JSON format.
    
    Args:
        file_paths: List of input file paths (TXT, JSON, CSV, PDF)
        output_path: Path to save the output JSON file
        source: Source identifier for metadata (default: "document")
        category: Category for metadata (default: "General")
    """
    output_data = []
    doc_id = 1

    for file_path in file_paths:
        # Extract text chunks
        text_chunks = extract_text_from_file(file_path)
        
        for text in text_chunks:
            # Generate keywords
            keywords = extract_keywords(text)
            
            # Create document entry
            doc_entry = {
                "id": str(doc_id),
                "text": text,
                "metadata": {
                    "source": source,
                    "category": category,
                    "keywords": keywords
                }
            }
            output_data.append(doc_entry)
            doc_id += 1

    # Save to output JSON file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)
        logger.info(f"Output saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving output to {output_path}: {str(e)}")

if __name__ == "__main__":
    # Example usage
    input_files = [
        "/home/kaustav/AIML/NLP/chatbot/4adb967f-cef4-44ee-b115-21a30b5f04c0.pdf"
    ]
    # Filter out non-existent files
    input_files = [f for f in input_files if os.path.exists(f)]
    
    if not input_files:
        logger.error("No valid input files provided.")
    else:
        convert_to_doc1_format(
            file_paths=input_files,
            output_path="converted_doc1.txt",
            source="converted_document",
            category="General"
        )