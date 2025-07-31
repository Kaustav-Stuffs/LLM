import platform
from typing import Dict, Any

def get_device_info() -> str:
    """Get information about the current device and available hardware."""
    device_info = []
    
    # CPU info
    device_info.append(f"CPU: {platform.processor() or platform.machine() or 'Unknown'}")
    
    # Platform info
    device_info.append(f"OS: {platform.system()}")
    
    # Python version
    device_info.append(f"Python: {platform.python_version()}")
    
    return " | ".join(device_info)

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length with suffix."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    import re
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters that might interfere with search
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', ' ', text)
    
    return text.strip()

def extract_keywords(text: str, min_length: int = 3) -> list:
    """Extract potential keywords from text."""
    import re
    
    # Simple keyword extraction
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Filter out short words and common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we',
        'they', 'me', 'him', 'her', 'us', 'them'
    }
    
    keywords = [word for word in words if len(word) >= min_length and word not in stop_words]
    
    # Return unique keywords
    return list(set(keywords))

def validate_document_structure(doc: Dict[str, Any]) -> bool:
    """Validate that a document has the required structure."""
    required_fields = ['text']
    optional_fields = ['id', 'metadata']
    
    # Check required fields
    for field in required_fields:
        if field not in doc:
            return False
        if not isinstance(doc[field], str) or not doc[field].strip():
            return False
    
    # Validate optional fields if present
    if 'id' in doc and not isinstance(doc['id'], (str, int)):
        return False
    
    if 'metadata' in doc and not isinstance(doc['metadata'], dict):
        return False
    
    return True

def get_document_stats(documents: list) -> Dict[str, Any]:
    """Get statistics about a collection of documents."""
    if not documents:
        return {
            "total_documents": 0,
            "total_characters": 0,
            "total_words": 0,
            "average_length": 0,
            "file_types": {}
        }
    
    total_chars = 0
    total_words = 0
    file_types = {}
    
    for doc in documents:
        if 'text' in doc:
            text = doc['text']
            total_chars += len(text)
            total_words += len(text.split())
        
        if 'metadata' in doc and 'file_type' in doc['metadata']:
            file_type = doc['metadata']['file_type']
            file_types[file_type] = file_types.get(file_type, 0) + 1
    
    return {
        "total_documents": len(documents),
        "total_characters": total_chars,
        "total_words": total_words,
        "average_length": total_chars / len(documents) if documents else 0,
        "file_types": file_types
    }

def create_document_id(filename: str, chunk_id: int) -> str:
    """Create a unique document ID."""
    import hashlib
    
    # Create a hash of the filename for uniqueness
    filename_hash = hashlib.md5(filename.encode()).hexdigest()[:8]
    return f"{filename_hash}_{chunk_id}"
