import fitz  # PyMuPDF
from pathlib import Path
import re
import traceback
from transformers import AutoTokenizer

class DocumentChunker:
    """
    Handles intelligent chunking of PDF documents into 512-1024 token chunks.
    Uses semantic boundaries and proper tokenization for optimal processing.
    """
    def __init__(self):
        # Initialize tokenizer for proper token counting
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        
        # Target chunk sizes in tokens
        self.min_chunk_tokens = 512
        self.max_chunk_tokens = 1024
        self.overlap_tokens = 50  # Overlap between chunks for context
        
        # Section title patterns for better structure detection
        self.title_patterns = [
            r"^(?:\d+\.?\s*)?([A-Z][^.!?\n]{15,80})(?:\.|\n|$)",
            r"^([A-Z][A-Z\s]{15,80})(?:\.|\n|$)",
            r"^([A-Z][^.!?\n]{15,80}:)",
            r"^((?:How to|Steps to|Guide to|Creating|Managing|Method \d+)\s[^.!?\n]{10,60})",
            r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){3,8})(?:\s*\([^)]+\))?(?:\.|\n|$)",
            r"^(Method \d+[^.!?\n]{5,60})(?:\.|\n|$)",
            r"^(Step \d+[^.!?\n]{5,60})(?:\.|\n|$)"
        ]

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the tokenizer."""
        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            return len(tokens)
        except:
            # Fallback to word count approximation
            return len(text.split()) * 1.3  # Rough approximation

    def extract_section_title(self, text: str, page_num: int, doc_name: str) -> str:
        """Extract meaningful section title from text."""
        # Try patterns first
        for pattern in self.title_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                title = match.group(1).strip()
                title = re.sub(r'^\W+|\W+$', '', title)
                title = re.sub(r'\s+', ' ', title)
                if 15 <= len(title) <= 80:
                    return title
        
        # Look for first meaningful sentence
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        for sentence in sentences[:2]:
            if (sentence and 
                15 <= len(sentence) <= 100 and
                len(sentence.split()) <= 12 and
                sentence[0].isupper()):
                
                if any(keyword in sentence.lower() for keyword in 
                      ['create', 'manage', 'fill', 'sign', 'form', 'document', 'pdf', 'method']):
                    return re.sub(r'[.,:;!?]$', '', sentence)
        
        # Fallback to first meaningful phrase
        words = text.split()[:8]
        if len(words) >= 4:
            title = ' '.join(words)
            title = re.sub(r'[.,:;!?]$', '', title)
            if len(title) <= 80:
                return title + "..."
        
        # Last resort
        doc_prefix = doc_name.replace('.pdf', '').replace('Learn Acrobat - ', '')
        return f"{doc_prefix} - Page {page_num}"

    def create_token_chunks(self, text: str, page_num: int, doc_name: str) -> list:
        """
        Split text into chunks of 512-1024 tokens with semantic boundaries.
        """
        chunks = []
        
        # Split by paragraphs first to maintain semantic boundaries
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        current_chunk = ""
        current_tokens = 0
        
        for paragraph in paragraphs:
            paragraph_tokens = self.count_tokens(paragraph)
            
            # If paragraph alone exceeds max tokens, split it by sentences
            if paragraph_tokens > self.max_chunk_tokens:
                sentences = [s.strip() + '.' for s in paragraph.split('.') if s.strip()]
                
                for sentence in sentences:
                    sentence_tokens = self.count_tokens(sentence)
                    
                    if current_tokens + sentence_tokens > self.max_chunk_tokens and current_chunk:
                        # Create chunk
                        if current_tokens >= self.min_chunk_tokens:
                            section_title = self.extract_section_title(current_chunk, page_num, doc_name)
                            chunks.append({
                                'text': current_chunk.strip(),
                                'tokens': current_tokens,
                                'section_title': section_title,
                                'page_number': page_num,
                                'document': doc_name
                            })
                        
                        # Start new chunk with overlap
                        overlap_words = current_chunk.split()[-self.overlap_tokens:]
                        current_chunk = ' '.join(overlap_words) + ' ' + sentence
                        current_tokens = self.count_tokens(current_chunk)
                    else:
                        current_chunk += ' ' + sentence
                        current_tokens += sentence_tokens
            
            # Normal paragraph processing
            elif current_tokens + paragraph_tokens > self.max_chunk_tokens and current_chunk:
                # Create chunk
                if current_tokens >= self.min_chunk_tokens:
                    section_title = self.extract_section_title(current_chunk, page_num, doc_name)
                    chunks.append({
                        'text': current_chunk.strip(),
                        'tokens': current_tokens,
                        'section_title': section_title,
                        'page_number': page_num,
                        'document': doc_name
                    })
                
                # Start new chunk with overlap
                overlap_words = current_chunk.split()[-self.overlap_tokens:]
                current_chunk = ' '.join(overlap_words) + ' ' + paragraph
                current_tokens = self.count_tokens(current_chunk)
            else:
                current_chunk += ' ' + paragraph
                current_tokens += paragraph_tokens
        
        # Add final chunk
        if current_chunk.strip() and current_tokens >= 100:  # Minimum viable chunk
            section_title = self.extract_section_title(current_chunk, page_num, doc_name)
            chunks.append({
                'text': current_chunk.strip(),
                'tokens': current_tokens,
                'section_title': section_title,
                'page_number': page_num,
                'document': doc_name
            })
        
        return chunks

    def extract_text_from_page(self, page):
        """Extract clean text from a PDF page."""
        try:
            # Try structured extraction first
            text_dict = page.get_text("dict")
            if "blocks" in text_dict:
                text_blocks = []
                for block in text_dict["blocks"]:
                    if block.get('type') == 0:  # text block
                        block_text = ""
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                block_text += span.get("text", "")
                        if block_text.strip():
                            text_blocks.append(block_text.strip())
                
                return '\n\n'.join(text_blocks)
        except:
            pass
        
        # Fallback to simple text extraction
        try:
            return page.get_text("text")
        except:
            return ""

    def process_document(self, pdf_path: Path) -> list:
        """
        Process a PDF document into 512-1024 token chunks.
        """
        chunks = []
        
        try:
            doc = fitz.open(pdf_path)
            doc_name = pdf_path.name
            
            print(f"Processing document: {doc_name}")
            
            for page_num in range(len(doc)):
                try:
                    page = doc[page_num]
                    page_text = self.extract_text_from_page(page)
                    
                    if page_text.strip():
                        # Create token-based chunks for this page
                        page_chunks = self.create_token_chunks(
                            page_text, page_num + 1, doc_name
                        )
                        chunks.extend(page_chunks)
                
                except Exception as e:
                    print(f"Error processing page {page_num + 1} in {doc_name}: {e}")
                    continue
            
            doc.close()
            print(f"Extracted {len(chunks)} chunks from {doc_name}")
            
        except Exception as e:
            print(f"Error opening document {pdf_path}: {e}")
            traceback.print_exc()
        
        return chunks

    def chunk_documents(self, pdf_paths: list) -> list:
        """
        Process multiple PDF documents and return all chunks with token counts.
        """
        all_chunks = []
        
        for pdf_path in pdf_paths:
            try:
                path_obj = Path(pdf_path) if isinstance(pdf_path, str) else pdf_path
                document_chunks = self.process_document(path_obj)
                all_chunks.extend(document_chunks)
                
            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")
                continue
        
        # Add chunk IDs and statistics
        for i, chunk in enumerate(all_chunks):
            chunk['chunk_id'] = i
        
        print(f"Successfully processed {len(pdf_paths)} documents into {len(all_chunks)} token-based chunks")
        
        # Print token statistics
        if all_chunks:
            token_counts = [chunk['tokens'] for chunk in all_chunks]
            print(f"Token stats - Min: {min(token_counts)}, Max: {max(token_counts)}, Avg: {sum(token_counts)/len(token_counts):.1f}")
        
        return all_chunks
