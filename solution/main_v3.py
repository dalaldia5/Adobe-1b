"""
Complete persona-driven document intelligence pipeline.
Uses MiniLM + T5-small + spaCy architecture for optimal performance.
"""

import json
import time
import os
from pathlib import Path
from typing import List, Dict, Any
import concurrent.futures

# Import our custom components
from chunker_v3 import TokenBasedChunker
from semantic_ranker import MiniLMSemanticRanker
from t5_summarizer_v2 import T5Summarizer

class PersonaDrivenDocumentIntelligence:
    """
    Complete ML pipeline using:
    - MiniLM-L6-v2 (~80MB, ~10ms per chunk) for semantic search
    - T5-small (~220MB, ~1s per 512-token chunk) for summarization
    - spaCy en_core_web_sm (~50MB) for NER and keywords
    - Token-based chunking (512-1024 tokens) for optimal processing
    """
    
    def __init__(self):
        print("Initializing Persona-Driven Document Intelligence Pipeline")
        print("Architecture: MiniLM + T5-small + spaCy")
        
        # Initialize pipeline components
        self.chunker = TokenBasedChunker()
        self.ranker = MiniLMSemanticRanker()
        self.summarizer = T5Summarizer()
        
        # Performance tracking
        self.start_time = None
        self.stats = {
            'documents_processed': 0,
            'chunks_created': 0,
            'chunks_ranked': 0,
            'chunks_summarized': 0,
            'total_time': 0
        }
        
        print("Pipeline ready!")
    
    def load_input_files(self, input_dir: str) -> tuple:
        """Load persona and job requirements from input files."""
        print("Loading input files...")
        
        job_file = os.path.join(input_dir, 'job.txt')
        persona_file = os.path.join(input_dir, 'persona.txt')
        
        try:
            with open(job_file, 'r', encoding='utf-8') as f:
                job = f.read().strip()
            
            with open(persona_file, 'r', encoding='utf-8') as f:
                persona = f.read().strip()
            
            print(f"Persona: {persona}")
            print(f"Job: {job}")
            
            return persona, job
            
        except Exception as e:
            print(f"Error loading input files: {e}")
            raise
    
    def process_documents(self, docs_dir: str, persona: str, job: str) -> List[Dict[str, Any]]:
        """
        Complete document processing pipeline:
        1. Token-based chunking (512-1024 tokens)
        2. MiniLM semantic ranking
        3. T5-small summarization
        """
        self.start_time = time.time()
        
        print(f"Processing documents from: {docs_dir}")
        print(f"Target: 5 PDFs in <60 seconds")
        
        # Step 1: Create chunks from all PDFs (parallelized)
        print("\nStep 1: Token-based chunking (parallel)...")
        all_chunks = []
        
        pdf_files = [f for f in os.listdir(docs_dir) if f.endswith('.pdf')]
        self.stats['documents_processed'] = len(pdf_files)
        
        print(f"Found {len(pdf_files)} PDF files")
        
        pdf_paths = [os.path.join(docs_dir, pdf_file) for pdf_file in pdf_files]
        
        # Parallelize PDF chunking
        with concurrent.futures.ThreadPoolExecutor() as executor:
            chunk_lists = list(executor.map(lambda path: self.chunker.process_pdf(Path(path)), pdf_paths))
        
        for pdf_file, chunks in zip(pdf_files, chunk_lists):
            print(f"   {pdf_file}: {len(chunks)} chunks")
            all_chunks.extend(chunks)
        
        self.stats['chunks_created'] = len(all_chunks)
        print(f"Created {len(all_chunks)} total chunks")
        
        # Step 2: Semantic ranking with MiniLM
        print("\nStep 2: MiniLM semantic ranking...")
        ranked_chunks = self.ranker.rank_chunks(
            all_chunks, persona, job, top_k=len(all_chunks)
        )
        
        self.stats['chunks_ranked'] = len(ranked_chunks)
        
        # Show diversity check
        doc_distribution = {}
        for chunk in ranked_chunks:
            doc = chunk['document']
            doc_distribution[doc] = doc_distribution.get(doc, 0) + 1
        print(f"Chunk distribution:")
        for doc, count in sorted(doc_distribution.items(), key=lambda x: x[1], reverse=True):
            print(f"   {doc}: {count} chunks")
        
        # Step 3: T5-small summarization (parallelized)
        print("\nStep 3: T5-small summarization (parallel)...")
        def summarize_chunk(chunk):
            return self.summarizer.summarize_chunk(chunk, persona, job)
        summarized_chunks = []
        # Use ThreadPoolExecutor for compatibility (avoid pickling issues)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            summaries = list(executor.map(summarize_chunk, ranked_chunks))
        for chunk, summary in zip(ranked_chunks, summaries):
            enhanced_chunk = chunk.copy()
            enhanced_chunk['summary'] = summary
            enhanced_chunk['summary_method'] = 'T5' if self.summarizer.model else 'extractive'
            summarized_chunks.append(enhanced_chunk)
        
        self.stats['chunks_summarized'] = len(summarized_chunks)
        
        # Step 4: Create final sections
        print("\nStep 4: Creating final sections...")
        sections = summarized_chunks  # Use summarized chunks as sections
        
        # Performance summary
        total_time = time.time() - self.start_time
        self.stats['total_time'] = total_time
        
        print(f"\nPerformance Summary:")
        print(f"   Total time: {total_time:.2f} seconds")
        print(f"   Documents: {self.stats['documents_processed']}")
        print(f"   Chunks created: {self.stats['chunks_created']}")
        print(f"   Chunks ranked: {self.stats['chunks_ranked']}")
        print(f"   Chunks summarized: {self.stats['chunks_summarized']}")
        print(f"   Final sections: {len(sections)}")
        print(f"   Speed: {self.stats['chunks_created']/total_time:.1f} chunks/sec")
        
        if total_time <= 60:
            print("Target achieved: <60 seconds!")
        else:
            print("Performance warning: >60 seconds")
        
        return sections
    
    def format_output(self, sections: List[Dict[str, Any]], persona: str, job: str) -> Dict[str, Any]:
        """Format results to match expected output structure in expectedOutput.json."""
        import datetime
        timestamp = datetime.datetime.now().isoformat()
        input_documents = list(set([section.get('document', 'Unknown') for section in sections]))
        # extracted_sections: only document, section_title, importance_rank, page_number
        extracted_sections = []
        subsection_analysis = []
        for i, section in enumerate(sections):
            extracted_sections.append({
                "document": section.get('document', 'Unknown'),
                "section_title": section.get('section_title', f'Section {i+1}'),
                "importance_rank": i + 1,
                "page_number": section.get('page_number', 1)
            })
            subsection_analysis.append({
                "document": section.get('document', 'Unknown'),
                "refined_text": section.get('summary', section.get('text', '')[:500] + "..."),
                "page_number": section.get('page_number', 1)
            })
        output = {
            "metadata": {
                "input_documents": input_documents,
                "persona": persona,
                "job_to_be_done": job,
                "processing_timestamp": timestamp
            },
            "extracted_sections": extracted_sections,
            "subsection_analysis": subsection_analysis
        }
        return output
    
    def save_output(self, output: Dict[str, Any], output_path: str):
        """Save results to JSON file."""
        print(f"Saving results to: {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print("Results saved successfully!")


def main():
    """Main execution function."""
    
    # Initialize pipeline
    pipeline = PersonaDrivenDocumentIntelligence()
    
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, '..', 'input')
    docs_dir = os.path.join(input_dir, 'docs')
    output_dir = os.path.join(base_dir, '..', 'output')
    output_file = os.path.join(output_dir, 'challenge_1b_output.json')
    
    try:
        # Load input requirements
        persona, job = pipeline.load_input_files(input_dir)
        
        # Process documents
        sections = pipeline.process_documents(docs_dir, persona, job)
        
        # Format and save output
        output = pipeline.format_output(sections, persona, job)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results
        pipeline.save_output(output, output_file)
        
        # Summary
        print(f"\nPipeline Complete!")
        print(f"Processed {pipeline.stats['documents_processed']} documents")
        print(f"Total time: {pipeline.stats['total_time']:.2f} seconds")
        print(f"Generated {len(sections)} relevant sections")
        
    except Exception as e:
        print(f"Pipeline error: {e}")
        raise


if __name__ == "__main__":
    main()
