from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch
from typing import List, Dict, Any
import re

class T5Summarizer:
    """
    Uses T5-small for high-quality summarization of relevant document chunks.
    Optimized for ~1s per 512-token chunk processing time.
    """
    
    def __init__(self):
        print("📝 Loading T5-small for summarization...")
        
        # Load T5-small model and tokenizer (~220MB)
        self.model = T5ForConditionalGeneration.from_pretrained('t5-small')
        self.tokenizer = AutoTokenizer.from_pretrained('t5-small')
        
        # Use CPU for compatibility (can be changed to GPU if available)
        self.device = "cpu"
        self.model.to(self.device)
        
        # Optimization settings for speed (~1s per chunk)
        self.max_input_length = 512
        self.max_output_length = 128
        self.num_beams = 2  # Reduced for speed
        
        print("✅ T5 summarizer ready!")
    
    def clean_text_for_t5(self, text: str) -> str:
        """Clean and prepare text for T5 processing."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might confuse T5
        text = re.sub(r'[^\w\s.,!?;:\-()]', ' ', text)
        
        # Ensure proper sentence structure
        text = text.strip()
        if not text.endswith(('.', '!', '?')):
            text += '.'
        
        return text
    
    def create_persona_prompt(self, text: str, persona: str, job: str) -> str:
        """Create T5 prompt optimized for persona-specific summarization."""
        # T5 works well with clear task descriptions
        prompt = (f"summarize the following text for a {persona} "
                 f"focusing on {job}: {text}")
        
        return prompt
    
    def summarize_chunk(self, text: str, persona: str, job: str) -> str:
        """
        Summarize a single chunk using T5-small (~1s processing time).
        """
        try:
            # Clean and prepare text
            clean_text = self.clean_text_for_t5(text)
            
            # Create persona-specific prompt
            prompt = self.create_persona_prompt(clean_text, persona, job)
            
            # Tokenize input
            inputs = self.tokenizer.encode(
                prompt,
                return_tensors="pt",
                max_length=self.max_input_length,
                truncation=True
            ).to(self.device)
            
            # Generate summary with optimized settings
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=self.max_output_length,
                    min_length=20,
                    num_beams=self.num_beams,
                    length_penalty=0.8,
                    early_stopping=True,
                    do_sample=False  # Deterministic for consistency
                )
            
            # Decode and clean summary
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            summary = summary.strip()
            
            # Post-process summary
            if len(summary) < 10:  # Fallback for very short summaries
                return self.extractive_fallback(clean_text)
            
            return summary
            
        except Exception as e:
            print(f"⚠️ T5 summarization error: {e}")
            return self.extractive_fallback(text)
    
    def extractive_fallback(self, text: str) -> str:
        """Simple extractive summarization fallback."""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if len(sentences) <= 2:
            return text[:200] + "..." if len(text) > 200 else text
        
        # Take first and most informative sentences
        summary_sentences = sentences[:2]
        return '. '.join(summary_sentences) + '.'
    
    def summarize_top_chunks(self, 
                           ranked_chunks: List[Dict[str, Any]], 
                           persona: str, 
                           job: str,
                           max_chunks: int = 10) -> List[Dict[str, Any]]:
        """
        Summarize the top-ranked chunks using T5-small.
        """
        print(f"📝 Summarizing top {min(max_chunks, len(ranked_chunks))} chunks with T5...")
        
        summarized_chunks = []
        
        for i, chunk in enumerate(ranked_chunks[:max_chunks]):
            print(f"   Processing chunk {i+1}/{min(max_chunks, len(ranked_chunks))}...")
            
            # Create copy to avoid modifying original
            chunk_copy = chunk.copy()
            
            # Generate summary
            summary = self.summarize_chunk(
                chunk['text'], 
                persona, 
                job
            )
            
            # Add summary to chunk
            chunk_copy['summary'] = summary
            chunk_copy['original_length'] = len(chunk['text'])
            chunk_copy['summary_length'] = len(summary)
            chunk_copy['compression_ratio'] = len(summary) / len(chunk['text'])
            
            summarized_chunks.append(chunk_copy)
        
        print(f"✅ Summarization complete!")
        return summarized_chunks
    
    def create_section_summaries(self, 
                                summarized_chunks: List[Dict[str, Any]], 
                                persona: str,
                                job: str) -> List[Dict[str, Any]]:
        """
        Create final section summaries for output.
        """
        print("📋 Creating final section summaries...")
        
        sections = []
        
        # Group chunks by document for diversity
        docs_processed = set()
        
        for chunk in summarized_chunks[:10]:  # Limit to top 10 for final output
            doc = chunk['document']
            
            # Create section entry
            section = {
                'document': doc,
                'page_number': chunk['page_number'],
                'section_title': self.generate_section_title(chunk, persona, job),
                'summary': chunk['summary'], 
                'relevance_score': chunk['relevance_score'],
                'token_count': chunk.get('token_count', 0),
                'keywords': chunk.get('keywords', [])[:5]  # Top 5 keywords
            }
            
            sections.append(section)
            docs_processed.add(doc)
        
        # Rank sections by relevance
        sections.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Add importance ranking
        for i, section in enumerate(sections):
            section['importance_rank'] = i + 1
        
        print(f"📊 Created {len(sections)} section summaries")
        print(f"📚 Documents covered: {len(docs_processed)}")
        
        return sections
    
    def generate_section_title(self, chunk: Dict[str, Any], persona: str, job: str) -> str:
        """Generate meaningful section title from chunk content."""
        text = chunk['text']
        
        # Try to extract a meaningful title from the first sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        for sentence in sentences[:3]:
            if (15 <= len(sentence) <= 80 and
                len(sentence.split()) <= 12 and
                any(keyword in sentence.lower() for keyword in 
                   ['create', 'manage', 'fill', 'sign', 'form', 'document', 'method', 'step'])):
                return sentence
        
        # Fallback: use T5 to generate a title
        try:
            title_prompt = f"generate a title for this content about {job}: {text[:200]}"
            inputs = self.tokenizer.encode(
                title_prompt,
                return_tensors="pt", 
                max_length=256,
                truncation=True
            )
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=50,
                    num_beams=2,
                    early_stopping=True
                )
            
            title = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if 10 <= len(title) <= 80:
                return title
                
        except:
            pass
        
        # Final fallback
        doc_name = chunk['document'].replace('.pdf', '').replace('Learn Acrobat - ', '')
        return f"{doc_name} - Page {chunk['page_number']} Content"
