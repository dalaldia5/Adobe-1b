from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import List, Dict, Any
import re

class T5Summarizer:
    """
    Uses FLAN-T5 small for high-quality summarization of relevant document chunks.
    Uses AutoTokenizer to avoid sentencepiece issues.
    """
    
    def __init__(self):
        print("Loading FLAN-T5-small for summarization...")
        
        try:
            # Try FLAN-T5 first (compatible with AutoTokenizer)
            self.model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-small')
            self.tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-small')
            print("Using FLAN-T5-small")
        except:
            try:
                # Fallback to regular T5
                self.model = AutoModelForSeq2SeqLM.from_pretrained('t5-small')
                self.tokenizer = AutoTokenizer.from_pretrained('t5-small')
                print("Using T5-small")
            except Exception as e:
                print(f"Error loading T5 models: {e}")
                print("Using fallback summarization method")
                self.model = None
                self.tokenizer = None
                return
        
        # Use CPU for compatibility
        self.device = "cpu"
        if self.model:
            self.model.to(self.device)
        
        # Optimization settings for speed
        self.max_input_length = 512
        self.max_output_length = 128
        self.num_beams = 2  # Fast beam search
        
        print("T5 Summarizer ready!")
    
    def create_persona_prompt(self, text: str, persona: str, job: str) -> str:
        """Create a persona-aware prompt for summarization."""
        # Create instruction for T5/FLAN-T5
        if "flan" in str(type(self.model)).lower():
            prompt = f"Summarize this text for a {persona} working on: {job}\n\nText: {text}\n\nSummary:"
        else:
            prompt = f"summarize: For {persona}: {text}"
        
        return prompt
    
    def summarize_chunk(self, chunk: Dict[str, Any], persona: str, job: str) -> str:
        """
        Generate persona-aware summary of a single chunk.
        Optimized for ~1s processing time per chunk.
        """
        if not self.model or not self.tokenizer:
            # Fallback to extractive summarization
            return self._extractive_fallback(chunk, persona, job)
        
        try:
            text = chunk.get('text', '')
            
            # Clean and truncate text
            text = self._clean_text(text)
            
            # Create persona-aware prompt
            prompt = self.create_persona_prompt(text, persona, job)
            
            # Tokenize with length limits
            inputs = self.tokenizer(
                prompt,
                max_length=self.max_input_length,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate summary
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=self.max_output_length,
                    min_length=20,
                    num_beams=self.num_beams,
                    early_stopping=True,
                    no_repeat_ngram_size=2,
                    do_sample=False  # Deterministic for consistency
                )
            
            # Decode summary
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up summary
            summary = self._post_process_summary(summary, persona, job)
            
            return summary
            
        except Exception as e:
            print(f"Error in T5 summarization: {e}")
            return self._extractive_fallback(chunk, persona, job)
    
    def summarize_chunks(self, chunks: List[Dict[str, Any]], persona: str, job: str) -> List[Dict[str, Any]]:
        """
        Generate summaries for multiple chunks efficiently.
        """
        print(f"Summarizing {len(chunks)} chunks with T5...")
        
        summarized_chunks = []
        
        for i, chunk in enumerate(chunks):
            print(f"   Processing chunk {i+1}/{len(chunks)}")
            
            summary = self.summarize_chunk(chunk, persona, job)
            
            # Create enhanced chunk with summary
            enhanced_chunk = chunk.copy()
            enhanced_chunk['summary'] = summary
            enhanced_chunk['summary_method'] = 'T5' if self.model else 'extractive'
            
            summarized_chunks.append(enhanced_chunk)
        
        print("Summarization complete!")
        return summarized_chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and prepare text for summarization."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove very short or empty content
        if len(text) < 50:
            return ""
        
        # Limit length to prevent token overflow
        if len(text) > 2000:
            text = text[:2000] + "..."
        
        return text
    
    def _post_process_summary(self, summary: str, persona: str, job: str) -> str:
        """Post-process the generated summary."""
        # Remove prompt remnants
        if summary.startswith("Summary:"):
            summary = summary[8:].strip()
        
        # Remove repetitive phrases
        summary = re.sub(r'\b(\w+\s+){3,}\1', r'\1', summary)
        
        # Ensure minimum length
        if len(summary) < 20:
            return f"Key information relevant to {persona} regarding {job}."
        
        # Capitalize first letter
        if summary and not summary[0].isupper():
            summary = summary[0].upper() + summary[1:]
        
        return summary.strip()
    
    def _extractive_fallback(self, chunk: Dict[str, Any], persona: str, job: str) -> str:
        """
        Fallback extractive summarization when T5 is not available.
        """
        text = chunk.get('text', '')
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if not sentences:
            return "No significant content found."
        
        # Score sentences based on keyword relevance
        persona_keywords = persona.lower().split()
        job_keywords = job.lower().split()
        all_keywords = persona_keywords + job_keywords
        
        scored_sentences = []
        for sentence in sentences:
            score = sum(keyword in sentence.lower() for keyword in all_keywords)
            scored_sentences.append((sentence, score))
        
        # Sort by score and take top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Select top sentences up to reasonable length
        selected = []
        total_length = 0
        for sentence, score in scored_sentences:
            if total_length + len(sentence) <= 200 and len(selected) < 3:
                selected.append(sentence)
                total_length += len(sentence)
            else:
                break
        
        if not selected:
            selected = [sentences[0]]  # At least one sentence
        
        return '. '.join(selected) + '.'

    def get_model_info(self) -> Dict[str, Any]:
        """Return information about the loaded model."""
        if not self.model:
            return {"model": "fallback", "status": "extractive_only"}
        
        return {
            "model": str(type(self.model)).split('.')[-1].replace("'>", ""),
            "tokenizer": str(type(self.tokenizer)).split('.')[-1].replace("'>", ""),
            "device": self.device,
            "status": "ready"
        }
