import fitz  # PyMuPDF
from pathlib import Path
import re
from typing import List, Dict, Any
from transformers import AutoTokenizer

class TokenBasedChunker:
    """
    Intelligent PDF chunker that creates 512-1024 token chunks using T5 tokenizer.
    Maintains semantic boundaries and proper context overlap.
    """
    
    def __init__(self):
        # Use AutoTokenizer instead of T5Tokenizer to handle dependencies better
        print("Loading T5 tokenizer for chunk sizing...")
        self.tokenizer = AutoTokenizer.from_pretrained('t5-small')
        
        # Chunk configuration 
        self.min_tokens = 512
        self.max_tokens = 1024
        self.overlap_tokens = 50
        
        print(f"Chunker initialized: {self.min_tokens}-{self.max_tokens} tokens per chunk")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using T5 tokenizer."""
        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            return len(tokens)
        except:
            # Fallback approximation
            return len(text.split()) 
    
    def extract_clean_text(self, page) -> str:
        """Extract clean text from PDF page."""
        try:
            # Get structured text
            text_dict = page.get_text("dict")
            blocks = []
            
            for block in text_dict.get("blocks", []):
                if block.get('type') == 0:  # text block
                    block_text = ""
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            block_text += span.get("text", "")
                    
                    if block_text.strip():
                        blocks.append(block_text.strip())
            
            return '\n\n'.join(blocks)
            
        except:
            # Fallback to simple extraction
            return page.get_text()
    
    def create_semantic_chunks(self, text: str, doc_name: str, page_num: int) -> List[Dict[str, Any]]:
        """
        Create 512-1024 token chunks with semantic boundaries.
        """
        chunks = []
        
        # Split by paragraphs to maintain semantic boundaries
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        current_chunk = ""
        current_tokens = 0
        
        for para in paragraphs:
            para_tokens = self.count_tokens(para)
            
            # If adding this paragraph exceeds max tokens
            if current_tokens + para_tokens > self.max_tokens and current_chunk:
                
                # Only create chunk if it meets minimum size
                if current_tokens >= self.min_tokens:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'token_count': current_tokens,
                        'document': doc_name,
                        'page_number': page_num,
                        'chunk_id': len(chunks)
                    })
                
                # Start new chunk with overlap
                words = current_chunk.split()
                if len(words) > self.overlap_tokens:
                    overlap = ' '.join(words[-self.overlap_tokens:])
                    current_chunk = overlap + ' ' + para
                else:
                    current_chunk = para
                
                current_tokens = self.count_tokens(current_chunk)
            
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += ' ' + para
                else:
                    current_chunk = para
                current_tokens += para_tokens
        
        # Add final chunk if it has content
        if current_chunk.strip() and current_tokens >= 200:  # Minimum viable chunk
            chunks.append({
                'text': current_chunk.strip(),
                'token_count': current_tokens,
                'document': doc_name,  
                'page_number': page_num,
                'chunk_id': len(chunks)
            })
        
        return chunks
    
    def process_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Process a single PDF into token-based chunks."""
        chunks = []
        
        try:
            doc = fitz.open(pdf_path)
            doc_name = pdf_path.name
            
            print(f"Processing: {doc_name}")
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = self.extract_clean_text(page)
                
                if page_text.strip():
                    page_chunks = self.create_semantic_chunks(
                        page_text, doc_name, page_num + 1
                    )
                    chunks.extend(page_chunks)
            
            doc.close()
            print(f"   Extracted {len(chunks)} chunks")
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
        
        return chunks
    
    def chunk_documents(self, pdf_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Process 3-10 PDF files into 512-1024 token chunks.
        """
        print(f"Chunking {len(pdf_paths)} documents...")
        
        all_chunks = []
        
        for pdf_path in pdf_paths:
            path_obj = Path(pdf_path)
            if path_obj.exists():
                doc_chunks = self.process_pdf(path_obj)
                all_chunks.extend(doc_chunks)
        
        # Add global chunk IDs
        for i, chunk in enumerate(all_chunks):
            chunk['global_chunk_id'] = i
        
        # Print statistics  
        if all_chunks:
            token_counts = [chunk['token_count'] for chunk in all_chunks]
            print(f"Chunk Statistics:")
            print(f"   Total chunks: {len(all_chunks)}")
            print(f"   Token range: {min(token_counts)}-{max(token_counts)}")  
            print(f"   Average: {sum(token_counts)//len(token_counts)} tokens")
        
        return all_chunks
