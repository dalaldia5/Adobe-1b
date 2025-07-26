import os
import sys
import json
from datetime import datetime
from typing import List, Dict, Any

# Add the solution directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Local imports
from chunker import DocumentChunker
from relevance_ranker import PersonaAwareRelevanceRanker

def main():
    """
    Main function for persona-driven document intelligence.
    Processes documents and extracts relevant sections based on persona and job requirements.
    """
    
    # Read input files
    input_dir = "input"
    docs_dir = os.path.join(input_dir, "docs")
    
    # Read persona and job descriptions
    try:
        with open(os.path.join(input_dir, "persona.txt"), "r", encoding="utf-8") as f:
            persona = f.read().strip()
    except FileNotFoundError:
        print("Error: persona.txt not found in input directory")
        return
    
    try:
        with open(os.path.join(input_dir, "job.txt"), "r", encoding="utf-8") as f:
            job = f.read().strip()
    except FileNotFoundError:
        print("Error: job.txt not found in input directory")
        return
    
    print(f"Persona: {persona}")
    print(f"Job: {job}")
    
    # Process exactly 5 PDF files for 60-second target
    pdf_files = [f for f in os.listdir(docs_dir) if f.endswith('.pdf')][:5]
    doc_paths = [os.path.join(docs_dir, f) for f in pdf_files]
    
    if not doc_paths:
        print("Error: No PDF documents found in input/docs directory")
        return
    
    print(f"Processing {len(doc_paths)} PDF documents in 60-second target")
    
    # Debug: Show which documents are being processed
    print("Documents:")
    for doc_path in doc_paths:
        print(f"  - {os.path.basename(doc_path)}")
    
    # Initialize components
    print("Initializing document chunker...")
    chunker = DocumentChunker()
    
    print("Initializing persona-aware relevance ranker...")
    ranker = PersonaAwareRelevanceRanker()
    
    # Process documents
    print("Processing documents...")
    chunks = chunker.chunk_documents(doc_paths)
    
    if not chunks:
        print("Error: No chunks extracted from documents")
        return
    
    print(f"Extracted {len(chunks)} chunks from documents")
    
    # Extract relevant sections based on persona and job
    print("Extracting relevant sections...")
    results = ranker.extract_sections_for_persona(chunks, persona, job)
    
    # Create output JSON
    output_data = {
        "metadata": {
            "input_documents": [os.path.basename(path) for path in doc_paths],
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": [],
        "subsection_analysis": []
    }
    
    # Add sections to output
    for i, section in enumerate(results['sections']):
        output_data["extracted_sections"].append({
            "document": section['document'],
            "page_number": section['page_number'],
            "section_title": section['section_title'],
            "importance_rank": i + 1
        })
    
    # Add subsections to output
    for subsection in results['subsections']:
        output_data["subsection_analysis"].append({
            "document": subsection['document'],
            "page_number": subsection['page_number'],
            "refined_text": subsection['refined_text']
        })
    
    # Save output
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "challenge_1b_output.json")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Output saved to {output_file}")
    print(f"Extracted {len(output_data['extracted_sections'])} sections and {len(output_data['subsection_analysis'])} subsections")

if __name__ == "__main__":
    main()