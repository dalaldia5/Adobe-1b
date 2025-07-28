"""
Download and cache all required models for the pipeline.
Run this script once before using the main pipeline.
"""

import os
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
import spacy
from spacy.cli import download as spacy_download

def download_miniLM():
    """Download MiniLM-L6-v2 model (~80MB)"""
    print("📥 Downloading MiniLM-L6-v2 model (~80MB)...")
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ MiniLM-L6-v2 downloaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error downloading MiniLM: {e}")
        return False

def download_t5_small():
    """Download T5-small model (~220MB)"""
    print("📥 Downloading T5-small model (~220MB)...")
    try:
        model = T5ForConditionalGeneration.from_pretrained('t5-small')
        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        print("✅ T5-small downloaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error downloading T5-small: {e}")
        return False

def download_spacy():
    """Download spaCy en_core_web_sm model (~50MB)"""
    print("📥 Downloading spaCy en_core_web_sm model (~50MB)...")
    try:
        # Try to load existing model first
        try:
            nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy model already available!")
            return True
        except OSError:
            # Download if not available
            spacy_download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy en_core_web_sm downloaded successfully!")
            return True
    except Exception as e:
        print(f"❌ Error downloading spaCy model: {e}")
        return False

def main():
    """Download all required models"""
    print("🚀 Downloading all models for the pipeline...")
    print("📊 Total size: ~350MB (MiniLM: 80MB + T5: 220MB + spaCy: 50MB)")
    
    success_count = 0
    
    # Download models
    if download_miniLM():
        success_count += 1
    
    if download_t5_small():
        success_count += 1
    
    if download_spacy():
        success_count += 1
    
    # Summary
    print(f"\n📈 Download Summary: {success_count}/3 models successful")
    
    if success_count == 3:
        print("🎉 All models downloaded successfully!")
        print("✅ Ready to run the main pipeline!")
    else:
        print("⚠️ Some models failed to download. Check your internet connection.")
        print("🔄 You can run this script again to retry failed downloads.")

if __name__ == "__main__":
    main()
