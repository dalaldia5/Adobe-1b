import fitz  # PyMuPDF
from pathlib import Path
import re
import traceback

class DocumentChunker:
    """
    Handles the intelligent chunking of PDF documents.
    This implementation chunks by paragraphs to maintain semantic context.
    Designed to work with diverse PDF types and formats across any domain.
    """
    def __init__(self):
        # Common section title patterns (generalized for any domain)
        self.title_patterns = [
            r"^(?:\d+\.?\s*)?([A-Z][^.!?\n]{10,60})(?:\.|\n|$)",  # Section with numbers like "1. Introduction to Forms"
            r"^([A-Z][A-Z\s]{10,60})(?:\.|\n|$)",  # ALL CAPS titles
            r"^([A-Z][^.!?\n]{10,60}:)",  # Title with colon
            r"^((?:How to|Steps to|Guide to|Creating|Managing|Working with|Using)\s[^.!?\n]{5,50})",  # Instructional titles
            r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){2,7})(?:\.|\n|$)",  # Multi-word proper titles
            r"^\s*([A-Z][^.!?\n]*(?:form|Form|PDF|document|field|sign|create|manage)[^.!?\n]*)(?:\.|\n|$)"  # Form-related titles
        ]
        
        # Keywords that might indicate important sections (domain-agnostic)
        self.important_keywords = [
            # Academic keywords
            "abstract", "introduction", "method", "methodology", "result", "conclusion",
            "discussion", "literature", "review", "analysis", "findings", "research",
            "experiment", "study", "hypothesis", "theory", "model", "algorithm",
            
            # Business keywords
            "summary", "executive", "overview", "strategy", "financial", "revenue",
            "performance", "market", "analysis", "recommendation", "investment",
            "growth", "risk", "opportunity", "competitive", "assessment",
            
            # Educational keywords
            "chapter", "lesson", "concept", "principle", "definition", "example",
            "exercise", "problem", "solution", "practice", "application", "skill",
            "learning", "objective", "outcome", "assessment", "evaluation",
            
            # General important sections
            "background", "objective", "purpose", "scope", "limitation", "implication",
            "recommendation", "future", "work", "reference", "bibliography", "appendix"
        ]
        
        # Minimum text length to be considered a valid chunk
        self.min_chunk_length = 30
        
        # Maximum number of retries for different extraction methods
        self.max_extraction_retries = 3

    def get_document_title(self, doc):
        """Extract the document title using multiple strategies."""
        # Strategy 1: Use metadata
        if doc.metadata and doc.metadata.get('title'):
            title = doc.metadata.get('title')
            if title and len(title.strip()) > 0:
                return title.strip()
        
        # Strategy 2: Look for large text on first page
        try:
            if len(doc) > 0:
                page = doc[0]
                # Try different text extraction methods
                for method in ["dict", "blocks", "text"]:
                    try:
                        if method == "dict":
                            text_data = page.get_text("dict")
                            if "blocks" in text_data:
                                # Find text with largest font size
                                max_size = 0
                                largest_text = ""
                                for block in text_data["blocks"]:
                                    if block.get('type') == 0:  # text block
                                        for line in block.get("lines", []):
                                            for span in line.get("spans", []):
                                                if span.get("size", 0) > max_size:
                                                    max_size = span.get("size", 0)
                                                    largest_text = span.get("text", "")
                                if largest_text:
                                    return largest_text.strip()
                        
                        elif method == "blocks":
                            # Try blocks method
                            blocks = page.get_text("blocks")
                            if blocks and len(blocks) > 0:
                                # First block might be the title
                                for block in blocks[:3]:  # Check first few blocks
                                    if isinstance(block, tuple) and len(block) > 1:
                                        text = block[4] if len(block) > 4 else block[1]
                                        if isinstance(text, str) and len(text.strip()) > 0:
                                            return text.strip()
                        
                        elif method == "text":
                            # Simple text extraction
                            text = page.get_text("text")
                            if text:
                                # Take first non-empty line
                                lines = [line.strip() for line in text.split('\n') if line.strip()]
                                if lines:
                                    return lines[0]
                    except Exception:
                        continue  # Try next method
        except Exception:
            pass
        
        # Fallback to filename
        return "Untitled Document"

    def extract_section_title(self, text, page_num, paragraph_num, doc_name):
        """
        Extract a meaningful section title from the paragraph text.
        Uses multiple strategies to find the best title.
        """
        # Strategy 1: Look for patterns that match typical section titles
        for pattern in self.title_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                title = match.group(1).strip()
                # Clean up the title
                title = re.sub(r'^\W+|\W+$', '', title)  # Remove leading/trailing punctuation
                title = re.sub(r'\s+', ' ', title)  # Normalize whitespace
                if 8 <= len(title) <= 80:  # Reasonable title length
                    return title
        
        # Strategy 2: Look for form-specific patterns
        form_patterns = [
            r"(create|creating|build|building|design|designing).*?(form|field|document)",
            r"(fill|filling|complete|completing).*?(form|field|document)",
            r"(manage|managing|organize|organizing).*?(form|document|data)",
            r"(sign|signing|signature|electronic signature)",
            r"(workflow|process|procedure).*?(form|document)"
        ]
        
        for pattern in form_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Extract a meaningful phrase around the match
                words = text.split()
                match_text = match.group(0).lower()
                for i, word in enumerate(words):
                    if any(w in word.lower() for w in match_text.split()):
                        start = max(0, i - 2)
                        end = min(len(words), i + 6)
                        title = ' '.join(words[start:end]).strip()
                        title = re.sub(r'[.,:;!?]$', '', title)  # Remove trailing punctuation
                        if 10 <= len(title) <= 80:
                            # Capitalize properly
                            return ' '.join(word.capitalize() if word.lower() not in ['of', 'and', 'the', 'in', 'with', 'for'] else word.lower() 
                                          for word in title.split())
        
        # Strategy 3: Look for instructional phrases that often indicate section titles
        first_sentence = text.split('.')[0] if '.' in text else text[:120]
        first_sentence = first_sentence.strip()
        
        # Check if the first sentence looks like a proper section title
        if (first_sentence and 
            len(first_sentence.split()) <= 15 and 
            10 <= len(first_sentence) <= 80 and
            first_sentence[0].isupper()):
            
            # Additional checks for quality titles
            if (any(keyword in first_sentence.lower() for keyword in 
                   ['create', 'manage', 'fill', 'sign', 'form', 'document', 'pdf', 'field', 'data']) or
                re.search(r'^\d+\.', first_sentence) or  # Numbered sections
                first_sentence.count(' ') >= 2):  # Multi-word titles
                
                return re.sub(r'[.,:;!?]$', '', first_sentence)
        
        # Strategy 4: Fallback - use first few words but make them more meaningful
        words = text.split()[:8]  # Take more words for better context
        if len(words) >= 3:
            title = ' '.join(words)
            title = re.sub(r'[.,:;!?]$', '', title)
            if len(title) <= 80:
                return title
        
        # Last resort - use document context
        doc_prefix = doc_name.replace('.pdf', '').replace('Learn Acrobat - ', '')
        return f"{doc_prefix} Section {paragraph_num}"

    def extract_text_with_fallbacks(self, page, method="dict"):
        """Extract text from a page with multiple fallback methods."""
        text_blocks = []
        
        # Try primary method first
        try:
            if method == "dict":
                text_dict = page.get_text("dict")
                if "blocks" in text_dict:
                    for block in text_dict["blocks"]:
                        if block.get('type') == 0:  # text block
                            block_text = ""
                            for line in block.get("lines", []):
                                for span in line.get("spans", []):
                                    block_text += span.get("text", "")
                            if block_text.strip():
                                text_blocks.append(block_text.strip())
            
            elif method == "blocks":
                blocks = page.get_text("blocks")
                for block in blocks:
                    if isinstance(block, tuple) and len(block) > 1:
                        text = block[4] if len(block) > 4 else block[1]
                        if isinstance(text, str) and text.strip():
                            text_blocks.append(text.strip())
            
            elif method == "text":
                text = page.get_text("text")
                paragraphs = text.split('\n\n')
                for para in paragraphs:
                    if para.strip():
                        text_blocks.append(para.strip())
                        
        except Exception:
            # If primary method fails, try fallbacks
            if method != "text":
                return self.extract_text_with_fallbacks(page, "text")
        
        return text_blocks

    def process_document(self, pdf_path: Path) -> list:
        """
        Opens a PDF and splits its content into paragraph-based chunks.
        Each chunk is augmented with document-level metadata.
        Handles diverse PDF formats with fallback strategies.
        """
        chunks = []
        try:
            doc = fitz.open(pdf_path)
            doc_title = self.get_document_title(doc)
            doc_name = pdf_path.name
            
            # Process each page
            for page_num in range(len(doc)):
                try:
                    page = doc[page_num]
                    page_num_1based = page_num + 1
                    
                    # Try different extraction methods
                    extraction_methods = ["dict", "blocks", "text"]
                    text_blocks = []
                    
                    for method in extraction_methods:
                        text_blocks = self.extract_text_with_fallbacks(page, method)
                        if text_blocks:
                            break
                    
                    # Process extracted blocks
                    for i, block_text in enumerate(text_blocks):
                        # Clean up text
                        paragraph_text = block_text.strip().replace('\n', ' ')
                        paragraph_text = re.sub(r'\s+', ' ', paragraph_text)
                        
                        # Only include substantive chunks
                        if len(paragraph_text) >= self.min_chunk_length:
                            # Extract a meaningful section title
                            section_title = self.extract_section_title(
                                paragraph_text, page_num_1based, i + 1, doc_name
                            )
                            
                            # Create chunk with metadata
                            chunk = {
                                "doc_name": doc_name,
                                "doc_title": doc_title,
                                "page_number": page_num_1based,  # Fixed field name
                                "content": paragraph_text,
                                "title": f"Page {page_num_1based}, Paragraph {i+1}",
                                "section_title": section_title
                            }
                            chunks.append(chunk)
                
                except Exception as e:
                    print(f"Error processing page {page_num} in {pdf_path.name}: {e}")
                    traceback.print_exc()
            
            doc.close()
        except Exception as e:
            print(f"Error opening document {pdf_path.name}: {e}")
            traceback.print_exc()
        
        return chunks
    
    def chunk_documents(self, pdf_paths: list) -> list:
        """
        Process multiple PDF documents and return all chunks.
        This is the main method called by the pipeline.
        """
        all_chunks = []
        
        for pdf_path in pdf_paths:
            try:
                # Convert string path to Path object if needed
                path_obj = Path(pdf_path) if isinstance(pdf_path, str) else pdf_path
                
                print(f"Processing document: {path_obj.name}")
                chunks = self.process_document(path_obj)
                
                # Add document name to each chunk
                for chunk in chunks:
                    chunk['doc_name'] = path_obj.name
                
                all_chunks.extend(chunks)
                
            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")
                continue
        
        print(f"Successfully processed {len(pdf_paths)} documents into {len(all_chunks)} chunks")
        return all_chunks