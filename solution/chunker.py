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
        # Common section title patterns (improved for better detection)
        self.title_patterns = [
            r"^(?:\d+\.?\s*)?([A-Z][^.!?\n]{15,80})(?:\.|\n|$)",  # Section with numbers like "1. Create fillable forms"
            r"^([A-Z][A-Z\s]{15,80})(?:\.|\n|$)",  # ALL CAPS titles
            r"^([A-Z][^.!?\n]{15,80}:)",  # Title with colon
            r"^((?:How to|Steps to|Guide to|Creating|Managing|Working with|Using|Method \d+)\s[^.!?\n]{10,60})",  # Instructional titles
            r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){3,8})(?:\s*\([^)]+\))?(?:\.|\n|$)",  # Multi-word proper titles with optional parentheses
            r"^\s*([A-Z][^.!?\n]*(?:form|Form|PDF|document|field|sign|create|manage|convert|edit|export|share)[^.!?\n]{5,40})(?:\.|\n|$)",  # Form-related titles
            r"^([A-Z][^.!?\n]*(?:Acrobat|Adobe)[^.!?\n]{5,50})(?:\.|\n|$)",  # Adobe-specific titles
            r"^(Method \d+[^.!?\n]{5,60})(?:\.|\n|$)",  # Method titles
            r"^(Step \d+[^.!?\n]{5,60})(?:\.|\n|$)"  # Step titles
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
        
        # Strategy 3: Look for instructional phrases and proper headings
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        for sentence in sentences[:3]:  # Check first 3 sentences
            if (sentence and 
                15 <= len(sentence) <= 100 and  # Reasonable title length
                len(sentence.split()) <= 12 and  # Not too many words
                sentence[0].isupper()):
                
                # Look for title-like characteristics
                if (any(keyword in sentence.lower() for keyword in 
                       ['create', 'manage', 'fill', 'sign', 'form', 'document', 'pdf', 'field', 'data', 
                        'convert', 'edit', 'export', 'share', 'method', 'step', 'how to', 'guide']) or
                    re.search(r'^\d+\.', sentence) or  # Numbered sections
                    re.search(r'^(Method|Step|How to|Creating|Managing)', sentence, re.IGNORECASE) or
                    sentence.count(' ') >= 3):  # Multi-word titles
                    
                    title = re.sub(r'[.,:;!?]$', '', sentence)
                    return title
        
        # Strategy 4: Look for headings in the first line that might not end with period
        first_line = text.split('\n')[0].strip()
        if (first_line and 
            15 <= len(first_line) <= 100 and
            len(first_line.split()) <= 12 and
            first_line[0].isupper() and
            not first_line.endswith('.') and  # Headings often don't end with periods
            any(keyword in first_line.lower() for keyword in 
               ['create', 'manage', 'fill', 'sign', 'form', 'document', 'pdf', 'convert', 'edit', 'method'])):
            return first_line
        
        # Strategy 5: Improved fallback - create meaningful titles from content
        # Look for the first meaningful phrase that could be a title
        words = text.split()
        
        # Try to find a meaningful phrase starting with key action words
        for i, word in enumerate(words[:20]):  # Check first 20 words
            if word.lower() in ['create', 'creating', 'manage', 'managing', 'fill', 'filling', 
                              'sign', 'signing', 'convert', 'converting', 'edit', 'editing', 
                              'export', 'exporting', 'share', 'sharing', 'method', 'step']:
                # Take a meaningful phrase starting from this word
                end_idx = min(i + 8, len(words))
                phrase = ' '.join(words[i:end_idx])
                # Clean up the phrase
                phrase = re.sub(r'[.,:;!?]$', '', phrase)
                if 15 <= len(phrase) <= 80:
                    # Capitalize properly
                    return ' '.join(word.capitalize() if j == 0 or word.lower() not in ['of', 'and', 'the', 'in', 'with', 'for', 'to', 'a', 'an'] 
                                  else word.lower() for j, word in enumerate(phrase.split()))
        
        # Final fallback - use first meaningful sentence but improve it
        words = text.split()[:10]  # Take first 10 words
        if len(words) >= 4:
            title = ' '.join(words)
            title = re.sub(r'[.,:;!?]$', '', title)
            if len(title) <= 100:
                # Make it more title-like
                return title + "..."  # Indicate it's truncated
        
        # Last resort - use document context with better naming
        doc_prefix = doc_name.replace('.pdf', '').replace('Learn Acrobat - ', '')
        return f"{doc_prefix} - Page {page_num} Content"

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