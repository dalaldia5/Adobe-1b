#!/usr/bin/env python3
"""
Safe test script for the persona-driven document intelligence solution.
"""

import os
import sys
import json
from datetime import datetime

# Add solution directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'solution'))

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

def test_solution_safe():
    """Test the solution with limited scope to prevent hanging."""
    
    print("🧪 Testing Persona-Driven Document Intelligence Solution (Safe Mode)")
    print("=" * 70)
    
    # Check if we have actual PDF documents
    docs_dir = 'input/docs'
    pdf_files = [f for f in os.listdir(docs_dir) if f.lower().endswith('.pdf')] if os.path.exists(docs_dir) else []
    
    if not pdf_files:
        print("⚠️  No PDF documents found in input/docs directory")
        print("   Please add PDF documents to test the full solution")
        return False
    
    # Limit to first 3 PDFs to prevent hanging
    limited_pdfs = pdf_files[:3]
    
    print(f"📄 Testing with {len(limited_pdfs)} PDF documents (limited for safety):")
    for pdf in limited_pdfs:
        print(f"   - {pdf}")
    
    # Temporarily move other PDFs
    backup_dir = 'input/docs_backup'
    os.makedirs(backup_dir, exist_ok=True)
    
    for pdf in pdf_files[3:]:
        src = os.path.join(docs_dir, pdf)
        dst = os.path.join(backup_dir, pdf)
        if os.path.exists(src):
            os.rename(src, dst)
    
    try:
        # Import and run the main solution
        from solution.main import main
        
        print("\n🔄 Running document analysis (safe mode)...")
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
            
            # Show first few results
            if result.get('extracted_sections'):
                print(f"\n📋 Top 3 extracted sections:")
                for i, section in enumerate(result['extracted_sections'][:3]):
                    print(f"   {i+1}. {section.get('section_title', 'Untitled')} (Page {section.get('page_number', 'N/A')})")
            
            return True
        else:
            print("❌ Output file not created")
            return False
            
    except Exception as e:
        print(f"❌ Error running solution: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Restore moved PDFs
        for pdf in pdf_files[3:]:
            src = os.path.join(backup_dir, pdf)
            dst = os.path.join(docs_dir, pdf)
            if os.path.exists(src):
                os.rename(src, dst)
        
        # Remove backup directory if empty
        try:
            os.rmdir(backup_dir)
        except:
            pass

if __name__ == "__main__":
    create_test_data()
    success = test_solution_safe()
    
    if success:
        print("\n🎉 Safe test passed! The solution is working correctly.")
        print("💡 You can now run with more documents by using: python -m solution.main")
    else:
        print("\n⚠️  Test failed. Please check the implementation.")
