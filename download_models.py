#!/usr/bin/env python3
"""
Model Downloads and Setup
Downloads and caches the required models for the persona-driven document intelligence system.
"""

import os
import sys
from pathlib import Path

def download_models():
    """Download and cache all required models."""
    
    print("🤖 Downloading and caching models...")
    print("=" * 50)
    
    # 1. Download MiniLM for semantic search (~80MB)
    print("📥 Downloading MiniLM-L6-v2 for semantic search...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print(f"✅ MiniLM-L6-v2 downloaded successfully (~80MB)")
    except Exception as e:
        print(f"❌ Error downloading MiniLM: {e}")
    
    # 2. Download T5-small for summarization (~220MB)  
    print("\n📥 Downloading T5-small for summarization...")
    try:
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        model = T5ForConditionalGeneration.from_pretrained('t5-small')
        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        print(f"✅ T5-small downloaded successfully (~220MB)")
    except Exception as e:
        print(f"❌ Error downloading T5-small: {e}")
    
    # 3. Download spaCy for NER/keywords (~50MB)
    print("\n📥 Downloading spaCy model for NER/keywords...")
    try:
        import spacy
        import subprocess
        
        # Download spaCy model via command line
        result = subprocess.run([
            sys.executable, "-m", "spacy", "download", "en_core_web_sm"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ spaCy en_core_web_sm downloaded successfully (~50MB)")
        else:
            print(f"⚠️ spaCy download warning: {result.stderr}")
            
        # Test loading
        nlp = spacy.load("en_core_web_sm")
        print(f"✅ spaCy model loaded and tested")
        
    except Exception as e:
        print(f"❌ Error downloading spaCy: {e}")
    
    # 4. Download basic tokenizer for chunk sizing
    print("\n📥 Downloading tokenizer for chunk sizing...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('t5-small')
        print(f"✅ T5-small tokenizer downloaded successfully")
    except Exception as e:
        print(f"❌ Error downloading tokenizer: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Model download complete!")
    print("Total approximate size: ~350MB")
    print("All models are cached for offline use.")

if __name__ == "__main__":
    download_models()
