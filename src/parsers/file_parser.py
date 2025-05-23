import os
from typing import Optional, List
from PyPDF2 import PdfReader

class FileParser:
    def __init__(self):
        self.supported_extensions = ['.txt', '.pdf']
    
    def read_file(self, file_path: str) -> Optional[str]:
        """
        Read a text or PDF file and return its contents.
        
        Args:
            file_path (str): Path to the text or PDF file
            
        Returns:
            Optional[str]: The content of the file if successful, None otherwise
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in self.supported_extensions:
                raise ValueError(f"Unsupported file extension: {file_ext}")

            if file_ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
            elif file_ext == '.pdf':
                content = self._read_pdf(file_path)
            
            return content
            
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            return None
    
    def _read_pdf(self, file_path: str) -> str:
        """
        Read and extract text from a PDF file.
        
        Args:
            file_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text content from the PDF
        """
        try:
            reader = PdfReader(file_path)
            text_content = []
            
            for page in reader.pages:
                text_content.append(page.extract_text())

            return '\n'.join(text_content)
            
        except Exception as e:
            raise Exception(f"Error reading PDF file: {str(e)}")
    
    def process_text_for_tts(self, text: str) -> List[str]:
        """
        Process text content for TTS by splitting into manageable chunks.
        
        Args:
            text (str): The text content to process
            
        Returns:
            List[str]: List of processed text chunks
        """
        if text is None:
            return []

        text = ' '.join(text.split())
        
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        return sentences