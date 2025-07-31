import os
import json
import csv
import re
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

# PDF processing - disabled for offline mode
PDF_AVAILABLE = False
USE_PDFPLUMBER = False

# DOCX processing - disabled for offline mode
DOCX_AVAILABLE = False

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles processing of various document formats into searchable chunks."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def process_file(self, file_path: str, original_filename: str) -> List[Dict[str, Any]]:
        """
        Process a file and return a list of document chunks.
        
        Args:
            file_path: Path to the file to process
            original_filename: Original filename for metadata
            
        Returns:
            List of document dictionaries with text and metadata
        """
        try:
            file_extension = Path(original_filename).suffix.lower()
            
            if file_extension == '.txt':
                return self._process_txt(file_path, original_filename)
            elif file_extension == '.json':
                return self._process_json(file_path, original_filename)
            elif file_extension == '.csv':
                return self._process_csv(file_path, original_filename)
            elif file_extension == '.pdf':
                return self._process_pdf(file_path, original_filename)
            elif file_extension == '.docx':
                return self._process_docx(file_path, original_filename)
            else:
                logger.warning(f"Unsupported file type: {file_extension}")
                return []
                
        except Exception as e:
            logger.error(f"Error processing file {original_filename}: {str(e)}")
            return []
    
    def _process_txt(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        """Process a text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                return []
            
            chunks = self._create_chunks(content)
            return [
                {
                    'id': f"{filename}_{i}",
                    'text': chunk,
                    'metadata': {
                        'source': filename,
                        'chunk_id': i,
                        'file_type': 'txt'
                    }
                }
                for i, chunk in enumerate(chunks)
            ]
        except Exception as e:
            logger.error(f"Error processing TXT file {filename}: {str(e)}")
            return []
    
    def _process_json(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        """Process a JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            documents = []
            
            if isinstance(data, list):
                # Handle list of documents
                for i, item in enumerate(data):
                    if isinstance(item, dict):
                        text = self._extract_text_from_dict(item)
                        if text.strip():
                            documents.append({
                                'id': f"{filename}_{i}",
                                'text': text,
                                'metadata': {
                                    'source': filename,
                                    'item_id': i,
                                    'file_type': 'json',
                                    'original_data': item
                                }
                            })
                    else:
                        # Handle simple list items
                        text = str(item)
                        if text.strip():
                            documents.append({
                                'id': f"{filename}_{i}",
                                'text': text,
                                'metadata': {
                                    'source': filename,
                                    'item_id': i,
                                    'file_type': 'json'
                                }
                            })
            
            elif isinstance(data, dict):
                # Handle single document or structured data
                text = self._extract_text_from_dict(data)
                if text.strip():
                    chunks = self._create_chunks(text)
                    documents = [
                        {
                            'id': f"{filename}_{i}",
                            'text': chunk,
                            'metadata': {
                                'source': filename,
                                'chunk_id': i,
                                'file_type': 'json',
                                'original_data': data
                            }
                        }
                        for i, chunk in enumerate(chunks)
                    ]
            
            return documents
            
        except Exception as e:
            logger.error(f"Error processing JSON file {filename}: {str(e)}")
            return []
    
    def _process_csv(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        """Process a CSV file."""
        try:
            documents = []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                # Try to detect if there's a header
                sample = f.read(1024)
                f.seek(0)
                
                reader = csv.reader(f)
                rows = list(reader)
                
                if not rows:
                    return []
                
                # Use first row as headers if it looks like headers
                headers = rows[0] if rows else []
                data_rows = rows[1:] if len(rows) > 1 else []
                
                # Create document for header row
                header_text = "Headers: " + ", ".join(headers)
                documents.append({
                    'id': f"{filename}_headers",
                    'text': header_text,
                    'metadata': {
                        'source': filename,
                        'row_type': 'headers',
                        'file_type': 'csv'
                    }
                })
                
                # Process data rows
                for i, row in enumerate(data_rows):
                    if len(row) == len(headers):
                        # Create structured text from row
                        row_text = "; ".join([f"{headers[j]}: {row[j]}" for j in range(len(row)) if row[j].strip()])
                    else:
                        # Fallback to simple comma-separated values
                        row_text = ", ".join([cell for cell in row if cell.strip()])
                    
                    if row_text.strip():
                        documents.append({
                            'id': f"{filename}_row_{i}",
                            'text': row_text,
                            'metadata': {
                                'source': filename,
                                'row_id': i,
                                'file_type': 'csv'
                            }
                        })
            
            return documents
            
        except Exception as e:
            logger.error(f"Error processing CSV file {filename}: {str(e)}")
            return []
    
    def _process_pdf(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        """Process a PDF file."""
        if not PDF_AVAILABLE:
            logger.error("PDF processing libraries not available. Install PyPDF2 or pdfplumber.")
            return []
        
        try:
            text_content = ""
            
            if USE_PDFPLUMBER:
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        page_text = page.extract_text()
                        if page_text:
                            text_content += f"\n[Page {page_num + 1}]\n{page_text}\n"
            else:
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    for page_num, page in enumerate(pdf_reader.pages):
                        page_text = page.extract_text()
                        if page_text:
                            text_content += f"\n[Page {page_num + 1}]\n{page_text}\n"
            
            if not text_content.strip():
                return []
            
            chunks = self._create_chunks(text_content)
            return [
                {
                    'id': f"{filename}_{i}",
                    'text': chunk,
                    'metadata': {
                        'source': filename,
                        'chunk_id': i,
                        'file_type': 'pdf'
                    }
                }
                for i, chunk in enumerate(chunks)
            ]
            
        except Exception as e:
            logger.error(f"Error processing PDF file {filename}: {str(e)}")
            return []
    
    def _process_docx(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        """Process a DOCX file."""
        if not DOCX_AVAILABLE:
            logger.error("DOCX processing library not available. Install python-docx.")
            return []
        
        try:
            doc = Document(file_path)
            text_content = ""
            
            for para in doc.paragraphs:
                if para.text.strip():
                    text_content += para.text + "\n"
            
            # Also extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        if cell.text.strip():
                            row_text.append(cell.text.strip())
                    if row_text:
                        text_content += " | ".join(row_text) + "\n"
            
            if not text_content.strip():
                return []
            
            chunks = self._create_chunks(text_content)
            return [
                {
                    'id': f"{filename}_{i}",
                    'text': chunk,
                    'metadata': {
                        'source': filename,
                        'chunk_id': i,
                        'file_type': 'docx'
                    }
                }
                for i, chunk in enumerate(chunks)
            ]
            
        except Exception as e:
            logger.error(f"Error processing DOCX file {filename}: {str(e)}")
            return []
    
    def _extract_text_from_dict(self, data: Dict[str, Any]) -> str:
        """Extract meaningful text from a dictionary."""
        text_parts = []
        
        for key, value in data.items():
            if isinstance(value, str) and value.strip():
                text_parts.append(f"{key}: {value}")
            elif isinstance(value, (int, float)):
                text_parts.append(f"{key}: {value}")
            elif isinstance(value, list):
                if value and all(isinstance(item, str) for item in value):
                    text_parts.append(f"{key}: {', '.join(value)}")
                elif value:
                    text_parts.append(f"{key}: {str(value)}")
            elif isinstance(value, dict):
                nested_text = self._extract_text_from_dict(value)
                if nested_text:
                    text_parts.append(f"{key}: {nested_text}")
        
        return "; ".join(text_parts)
    
    def _create_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        if not text.strip():
            return []
        
        # Clean up the text
        text = re.sub(r'\s+', ' ', text).strip()
        
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Find the end position for this chunk
            end = start + self.chunk_size
            
            if end >= len(text):
                # Last chunk
                chunks.append(text[start:].strip())
                break
            
            # Try to break at a sentence or paragraph boundary
            break_point = self._find_break_point(text, start, end)
            
            chunk = text[start:break_point].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = break_point - self.chunk_overlap
            if start < 0:
                start = 0
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def _find_break_point(self, text: str, start: int, end: int) -> int:
        """Find a good break point for chunking."""
        # Look for sentence endings first
        for i in range(end - 1, start, -1):
            if text[i] in '.!?':
                # Make sure it's not an abbreviation
                if i + 1 < len(text) and text[i + 1].isspace():
                    return i + 1
        
        # Look for paragraph breaks
        for i in range(end - 1, start, -1):
            if text[i] == '\n':
                return i + 1
        
        # Look for any whitespace
        for i in range(end - 1, start, -1):
            if text[i].isspace():
                return i + 1
        
        # No good break point found, just use the end
        return end

    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        formats = ['txt', 'json', 'csv']
        
        if PDF_AVAILABLE:
            formats.append('pdf')
        
        if DOCX_AVAILABLE:
            formats.append('docx')
        
        return formats
