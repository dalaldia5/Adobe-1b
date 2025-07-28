#!/usr/bin/env python3
"""
Debug script to see what chunks are being extracted and why Fill and Sign isn't appearing.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'solution'))

from solution.chunker import DocumentChunker
from solution.relevance_ranker import PersonaAwareRelevanceRanker

def debug_chunks():
    # Initialize chunker
    chunker = DocumentChunker()
    
    # Process only Fill and Sign PDF
    fill_sign_path = "input/docs/Learn Acrobat - Fill and Sign.pdf"
    
    if not os.path.exists(fill_sign_path):
        print(f"File not found: {fill_sign_path}")
        return
    
    print("Processing Fill and Sign PDF...")
    chunks = chunker.process_document(fill_sign_path)
    
    print(f"Extracted {len(chunks)} chunks from Fill and Sign PDF")
    
    # Show first 5 chunks
    for i, chunk in enumerate(chunks[:5]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Document: {chunk.get('document', 'N/A')}")
        print(f"Page: {chunk.get('page_number', 'N/A')}")
        print(f"Section Title: {chunk.get('section_title', 'N/A')}")
        print(f"Text (first 100 chars): {chunk.get('text', '')[:100]}...")
    
    # Test relevance scoring
    print("\n=== Testing Relevance Scoring ===")
    ranker = PersonaAwareRelevanceRanker()
    
    persona = "HR professional"
    job = "Create and manage fillable forms for onboarding and compliance"
    
    # Score the first few chunks
    for i, chunk in enumerate(chunks[:3]):
        score = ranker.calculate_relevance_score(chunk, persona, job)
        print(f"\nChunk {i+1} Relevance Score: {score:.4f}")
        print(f"Text: {chunk.get('text', '')[:150]}...")

if __name__ == "__main__":
    debug_chunks()
