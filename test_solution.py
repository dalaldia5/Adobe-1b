#!/usr/bin/env python3
"""
Test script to verify the persona-driven document intelligence solution works correctly.
"""

import os
import sys
import json
from datetime import datetime

# Add solution directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'solution'))

from solution.main import main

def create_test_data():
    """Create minimal test data to verify the solution works."""
    
    # Create input directory structure
    os.makedirs('input', exist_ok=True)
    os.makedirs('input/docs', exist_ok=True)
    
    # Create test persona
    with open('input/persona.txt', 'w', encoding='utf-8') as f:
        f.write("PhD Researcher in Computational Biology with expertise in machine learning and data analysis")
    
    # Create test job description
    with open('input/job.txt', 'w', encoding='utf-8') as f:
        f.write("Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks for drug discovery research")
    
    print("✓ Test data created successfully")

def test_solution():
    """Test the solution with sample data."""
    
    print("🧪 Testing Persona-Driven Document Intelligence Solution")
    print("=" * 60)
    
    # Check if we have actual PDF documents
    docs_dir = 'input/docs'
    pdf_files = [f for f in os.listdir(docs_dir) if f.lower().endswith('.pdf')] if os.path.exists(docs_dir) else []
    
    if not pdf_files:
        print("⚠️  No PDF documents found in input/docs directory")
        print("   Please add PDF documents to test the full solution")
        return False
    
    print(f"📄 Found {len(pdf_files)} PDF documents:")
    for pdf in pdf_files:
        print(f"   - {pdf}")
    
    try:
        # Run the main solution
        print("\n🔄 Running document analysis...")
        main()
        
        # Check if output was created
        if os.path.exists('output/challenge_1b_output.json'):
            with open('output/challenge_1b_output.json', 'r', encoding='utf-8') as f:
                result = json.load(f)
            
            print("\n✅ Solution completed successfully!")
            print(f"📊 Results:")
            print(f"   - Extracted {len(result.get('extracted_sections', []))} sections")
            print(f"   - Generated {len(result.get('sub_section_analysis', []))} subsection summaries")
            print(f"   - Processing timestamp: {result.get('metadata', {}).get('processing_timestamp', 'N/A')}")
            
            return True
        else:
            print("❌ Output file not created")
            return False
            
    except Exception as e:
        print(f"❌ Error running solution: {e}")
        return False

if __name__ == "__main__":
    create_test_data()
    success = test_solution()
    
    if success:
        print("\n🎉 All tests passed! The solution is ready for deployment.")
    else:
        print("\n⚠️  Some tests failed. Please check the implementation.")
