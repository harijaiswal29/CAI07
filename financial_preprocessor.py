import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import re
from bs4 import BeautifulSoup
import PyPDF2
from pathlib import Path
import pdfplumber
from typing import List, Generator
import re


# Periods that are NOT sentence terminators — protected with <DOT> before splitting,
# restored afterwards. Without this, "$309.1 billion" and "Alphabet Inc. The..." both
# split mid-token and leave chunks like "1 billion for 2023..." that downstream T5 can't fix.
_MULTI_DOT_ABBR_RE = re.compile(r'\b(U\.S|U\.K|e\.g|i\.e)\.')
_SINGLE_DOT_ABBR_RE = re.compile(
    r'\b(Inc|Corp|Ltd|Co|LLC|LLP|Plc|Mr|Mrs|Ms|Dr|Jr|Sr|St|No|vs|etc|Fig|Eq|Sec)\.'
)
_DECIMAL_RE = re.compile(r'(\d)\.(\d)')
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


def _protect_periods(text: str) -> str:
    text = _MULTI_DOT_ABBR_RE.sub(lambda m: m.group(1).replace('.', '<DOT>') + '<DOT>', text)
    text = _SINGLE_DOT_ABBR_RE.sub(lambda m: m.group(1) + '<DOT>', text)
    text = _DECIMAL_RE.sub(r'\1<DOT>\2', text)
    return text


def _restore_periods(text: str) -> str:
    return text.replace('<DOT>', '.')


class FinancialDataPreprocessor:
    """
    A class to preprocess financial statements for RAG implementation
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        """
        Initialize the FinancialDataPreprocessor
        
        Args:
            chunk_size (int): Size of each text chunk (default: 500)
            chunk_overlap (int): Overlap between chunks (default: 50)
        """
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
            raise ValueError("chunk_overlap must be a non-negative integer")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
            
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def read_financial_statement(self, file_path: str) -> str:
        """
        Read financial statements from various formats (PDF, HTML, DOCX)
        """
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            # pdfplumber preserves word boundaries better than PyPDF2 — avoids
            # "ca sh" / "increase s" artifacts that polluted earlier chunks.
            return self._read_pdf_with_pdfplumber(file_path)
        elif file_ext == '.html':
            return self._read_html(file_path)
        elif file_ext in ['.doc', '.docx']:
            return self._read_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def _read_pdf(self, file_path: str) -> str:
    #Extract text from PDF files with progress tracking and optimization
        try:
            # Use a list for string concatenation instead of +=
            text_parts = []
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                print(f"\nProcessing PDF with {total_pages} pages...")
                
                # Process pages in batches for better performance
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        # Extract text from the current page
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                        
                        # Print progress every 10 pages
                        if page_num % 10 == 0:
                            print(f"Processed {page_num}/{total_pages} pages...")
                            
                    except Exception as e:
                        print(f"Warning: Could not read page {page_num}: {str(e)}")
                        continue
                
                print("PDF processing completed!")
                
            # Join all text parts at once (more efficient than += for strings)
            return '\n'.join(text_parts)
            
        except Exception as e:
            print(f"Error reading PDF {file_path}: {str(e)}")
            return ""
    
    def _read_pdf_with_pdfplumber(self, file_path: str) -> str:
    #Extract text from PDF files using pdfplumber
        text_parts = []        
        try:
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                print(f"\nProcessing PDF with {total_pages} pages...")
                
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        text = page.extract_text()
                        if text:
                            text_parts.append(text)
                        
                        if page_num % 10 == 0:
                            print(f"Processed {page_num}/{total_pages} pages...")
                            
                    except Exception as e:
                        print(f"Warning: Could not read page {page_num}: {str(e)}")
                        continue
                        
                print("PDF processing completed!")
                
            return '\n'.join(text_parts)
        
        except Exception as e:
            print(f"Error reading PDF {file_path}: {str(e)}")
            return ""

    def _read_html(self, file_path: str) -> str:
            """Extract text from HTML files"""
            with open(file_path, 'r') as file:
                soup = BeautifulSoup(file, 'html.parser')
                return soup.get_text()    
    
    def _read_docx(self, file_path: str) -> str:
            """Extract text from DOCX files"""
            import docx
            doc = docx.Document(file_path)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])    

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize financial text data
        """
        # Remove special characters and normalize whitespace
        text = re.sub(r'[\n\r\t]+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Standardize financial notation
        text = re.sub(r'[\$€£¥]', '', text)  # Remove currency symbols
        text = re.sub(r'[^\w\s.,()-]', '', text)  # Keep essential punctuation
        
        # Normalize numbers
        text = re.sub(r'(\d),(\d)', r'\1\2', text)  # Remove commas in numbers
        
        return text.strip()

    def extract_financial_data(self, text: str) -> Dict:
        """
        Extract key financial metrics and data points
        """
        financial_data = {
            'revenue': [],
            'expenses': [],
            'profit': [],
            'assets': [],
            'liabilities': [],
            'dates': []
        }
        
        # Extract dates
        dates = re.findall(r'\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4}', text)
        financial_data['dates'] = dates
        
        # Extract monetary values (assumes format like $1,234.56 or 1,234.56)
        money_pattern = r'\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?'
        
        # Basic financial metric extraction (can be enhanced based on specific needs)
        sections = text.split('\n')
        for section in sections:
            section = section.lower()
            if 'revenue' in section:
                values = re.findall(money_pattern, section)
                financial_data['revenue'].extend(values)
            elif 'expense' in section:
                values = re.findall(money_pattern, section)
                financial_data['expenses'].extend(values)
            elif 'profit' in section or 'net income' in section:
                values = re.findall(money_pattern, section)
                financial_data['profit'].extend(values)
            elif 'asset' in section:
                values = re.findall(money_pattern, section)
                financial_data['assets'].extend(values)
            elif 'liabilit' in section:
                values = re.findall(money_pattern, section)
                financial_data['liabilities'].extend(values)
                
        return financial_data

    def create_chunks(self, text: str) -> List[str]:
        """Sentence-aware chunking with real (non-discardable) overlap."""
        protected = _protect_periods(text)
        sentences = [
            _restore_periods(s).strip()
            for s in _SENTENCE_SPLIT_RE.split(protected)
            if s.strip()
        ]

        print(f"\nProcessing text of length: {len(text)} characters ({len(sentences)} sentences)")

        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            if current and current_len + sentence_len + 1 > self.chunk_size:
                chunks.append(' '.join(current))

                # Carry trailing sentences whose cumulative length first reaches chunk_overlap.
                # Always keep at least the last sentence, even if it exceeds chunk_overlap on its own.
                overlap: List[str] = []
                overlap_len = 0
                for s in reversed(current):
                    overlap.insert(0, s)
                    overlap_len += len(s) + 1
                    if overlap_len >= self.chunk_overlap:
                        break
                current = overlap
                current_len = sum(len(s) + 1 for s in current)

            current.append(sentence)
            current_len += sentence_len + 1

        if current:
            chunks.append(' '.join(current))

        print(f"Chunking completed! Created {len(chunks)} chunks")
        return chunks

    def process_financial_statements(self, file_paths: List[str]) -> Tuple[List[str], Dict]:
        """
        Main method to process multiple financial statements
        """
        all_chunks = []
        all_financial_data = {}
        
        for file_path in file_paths:
            # Read the document
            text = self.read_financial_statement(file_path)
            
            # Clean the text
            cleaned_text = self.clean_text(text)
            
            # Extract financial data
            financial_data = self.extract_financial_data(cleaned_text)
            
            # Create chunks
            chunks = self.create_chunks(cleaned_text)
            
            # Store results
            all_chunks.extend(chunks)
            all_financial_data[file_path] = financial_data
            
        return all_chunks, all_financial_data

    def analyze_chunks(self, chunks: List[str]) -> Dict:
        """
        Analyze chunks for quality and coverage
        """
        analysis = {
            'num_chunks': len(chunks),
            'avg_chunk_length': np.mean([len(chunk) for chunk in chunks]),
            'min_chunk_length': min(len(chunk) for chunk in chunks),
            'max_chunk_length': max(len(chunk) for chunk in chunks),
            'total_tokens': sum(len(chunk.split()) for chunk in chunks)
        }
        return analysis

# Example usage
def main():
    # Initialize preprocessor
    preprocessor = FinancialDataPreprocessor(chunk_size=500, chunk_overlap=50)
    
    # Example file paths (replace with actual paths)
    file_paths = [
        'fin_stmts/goog-10-k-2024.pdf'
    ]
    
    # Process financial statements
    chunks, financial_data = preprocessor.process_financial_statements(file_paths)
    
    # Analyze chunks
    chunk_analysis = preprocessor.analyze_chunks(chunks)
    
    # Print results
    print("Preprocessing Complete!")
    print(f"Number of chunks created: {chunk_analysis['num_chunks']}")
    print(f"Average chunk length: {chunk_analysis['avg_chunk_length']:.2f} characters")
    print(f"Total tokens: {chunk_analysis['total_tokens']}")
    
    return chunks, financial_data, chunk_analysis

if __name__ == "__main__":
    try:
        main()  # This will run the full processing pipeline
    except Exception as e:
        print(f"Error: {str(e)}")
