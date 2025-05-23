import os
import re
import PyPDF2
from pathlib import Path
from typing import List, Optional

class FileParser:
    """Text file parsing utilities for TTS processing."""
    
    def __init__(self, min_chunk_length: int = 10, max_chunk_length: int = 500):
        """
        Initialize the parser with chunk size limits.
        
        Args:
            min_chunk_length: Minimum text chunk length to process
            max_chunk_length: Maximum text chunk length to process
        """
        self.min_chunk_length = min_chunk_length
        self.max_chunk_length = max_chunk_length
    
    def read_file(self, file_path: str) -> Optional[str]:
        """
        Read file content based on file extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Optional[str]: File content or None if file not found or not supported
        """
        try:
            if not os.path.exists(file_path):
                print(f"Error reading file: File not found: {file_path}")
                return None
            
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.pdf':
                return self._read_pdf(file_path)
            elif file_ext == '.docx':
                return self._read_docx(file_path)
            elif file_ext in ['.txt', '.md', '.rst', '.py', '.html', '.css', '.js', '.json']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                print(f"Unsupported file format: {file_ext}")
                return None
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            return None
    
    def _read_pdf(self, file_path: str) -> str:
        """Extract text from PDF file."""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def _read_docx(self, file_path: str) -> str:
        """Extract text from DOCX file."""
        doc = docx.Document(file_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    
    def process_text_for_tts(self, text: str) -> List[str]:
        """
        Process text into manageable chunks for TTS.
        
        Args:
            text: The input text to process
            
        Returns:
            List[str]: List of text chunks optimized for TTS
        """
        if not text:
            return []
            
        paragraphs = re.split(r'\n+', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        for para in paragraphs:
            if len(para) <= self.max_chunk_length:
                if len(para) >= self.min_chunk_length:
                    chunks.append(para)
            else:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) + 1 <= self.max_chunk_length:
                        if current_chunk:
                            current_chunk += " " + sentence
                        else:
                            current_chunk = sentence
                    else:
                        if current_chunk and len(current_chunk) >= self.min_chunk_length:
                            chunks.append(current_chunk)
                        current_chunk = sentence
                
                if current_chunk and len(current_chunk) >= self.min_chunk_length:
                    chunks.append(current_chunk)
                    
        return chunks