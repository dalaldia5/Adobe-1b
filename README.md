# Persona-Driven Document Intelligence Pipeline

A high-performance ML pipeline that extracts and summarizes the most relevant content from Adobe Acrobat documentation based on user personas and job requirements.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CPU-only environment (no GPU required)
- ~1GB disk space for models

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd adobe-1b

# Install dependencies
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm
```

### Usage
```bash
# Run the complete pipeline
python solution/main_v3.py
```

The pipeline will:
1. Load persona and job requirements from `input/` folder
2. Process all PDFs in `input/docs/` folder
3. Generate results in `output/challenge_1b_output.json`

## 📊 Performance Metrics

- **Processing Time**: ~24 seconds for 5 PDFs ✅
- **Model Size**: ~350MB total ✅
- **CPU Only**: No GPU required ✅
- **Offline**: No internet access needed ✅

## 🏗️ Architecture

### Core Components
- **MiniLM-L6-v2** (~80MB): Semantic search and relevance ranking
- **FLAN-T5-small** (~220MB): Persona-aware text summarization  
- **spaCy** (~50MB): Named entity recognition and keyword extraction
- **Token-based Chunking**: Precise 512-1024 token chunks for optimal processing

### Processing Pipeline
```
PDFs → Token Chunking → Semantic Ranking → T5 Summarization → Structured Output
```

## 📁 Project Structure

```
adobe-1b/
├── solution/
│   ├── main_v3.py              # Main pipeline orchestrator
│   ├── chunker_v3.py           # Token-based document chunking
│   ├── semantic_ranker.py      # MiniLM-based relevance ranking
│   ├── t5_summarizer_v2.py     # FLAN-T5 summarization
│   └── __init__.py
├── input/
│   ├── job.txt                 # Job requirements
│   ├── persona.txt             # User persona
│   └── docs/                   # PDF documents to process
├── output/
│   └── challenge_1b_output.json # Generated results
├── requirements.txt            # Python dependencies
├── explanation.md              # Detailed technical explanation
└── README.md                   # This file
```

## 📝 Input Format

### Persona File (`input/persona.txt`)
```

```

### Job Requirements (`input/job.txt`)
```

```


## 📤 Output Format

```json
{
  "metadata": {
    "input_documents": ["document1.pdf", "document2.pdf"],
    "persona": "HR professional",
    "job_to_be_done": "Create and manage fillable forms...",
    "processing_timestamp": "2025-07-28T16:52:20.941307",
    "processing_stats": {
      "total_documents": 5,
      "total_chunks_created": 61,
      "total_processing_time": "24.51s",
      "constraints_met": {
        "cpu_only": true,
        "model_size_under_1gb": true,
        "processing_under_60s": true,
        "no_internet_access": true
      }
    }
  },
  "extracted_sections": [
    {
      "document": "Learn Acrobat - Fill and Sign.pdf",
      "page_number": 15,
      "section_title": "Section 1",
      "importance_rank": 1,
      "sub_section_analysis": {
        "refined_text": "Open a PDF document. Sign the document...",
        "relevance_score": 1.417,
        "keywords": ["signature", "form", "field", "interactive"],
        "token_count": 466
      }
    }
  ]
}
```

## 🎯 Key Features

### Persona-Aware Processing
- Dynamic content relevance based on user role
- Task-specific prioritization
- Context-aware summarization

### Fill & Sign Content Boosting
- Special emphasis on form-related content
- Enhanced scoring for HR onboarding workflows
- Signature and form field prioritization

### Performance Optimized
- Token-precise chunking prevents overflow
- Efficient batch processing
- CPU-optimized model selection
- Focuses on top 7 most relevant chunks

### Robust & Reliable
- Fallback mechanisms for model failures
- Multiple PDF extraction methods
- Comprehensive error handling
- Deterministic output

## 🔧 Configuration

### Adjusting Number of Sections
To change from 7 to a different number of sections, modify these values in `main_v3.py`:

```python
# Line ~100: Semantic ranking
ranked_chunks = self.ranker.rank_chunks(all_chunks, persona, job, top_k=7)

# Line ~110: Summarization  
top_chunks = ranked_chunks[:7]

# Line ~170: Final output
top_sections = sections[:7]
```

### Model Customization
Models can be swapped by modifying the respective component files:
- MiniLM model: `semantic_ranker.py`
- T5 model: `t5_summarizer_v2.py`
- spaCy model: `semantic_ranker.py`

## 🐛 Troubleshooting

### Common Issues

**1. sentencepiece Installation Error**
```bash
# Use pre-built wheels
pip install --only-binary=all sentencepiece
```

**2. spaCy Model Missing**
```bash
python -m spacy download en_core_web_sm
```

**3. Memory Issues**
- Reduce chunk size in `chunker_v3.py`
- Decrease `top_k` value in `main_v3.py`

**4. Slow Processing**
- Check CPU usage and available memory
- Ensure no other heavy processes are running

### Debug Mode
Add debug prints by uncommenting relevant sections in the main pipeline or set logging level:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 Dependencies

### Core ML Libraries
- `torch>=1.9.0`: PyTorch for model inference
- `transformers>=4.30.0`: Hugging Face transformers
- `sentence-transformers>=2.2.0`: MiniLM embeddings
- `spacy>=3.7.0`: NLP and NER

### Document Processing
- `pymupdf>=1.23.0`: PDF text extraction
- `tokenizers>=0.13.0`: Token counting

### Utilities
- `numpy>=1.24.0`: Numerical computations
- `scikit-learn>=1.3.0`: Similarity calculations
- `tqdm>=4.65.0`: Progress tracking

## 🚧 Development

### Running Tests
```bash
# Basic functionality test
python solution/main_v3.py

# Check model loading
python -c "from solution.semantic_ranker import MiniLMSemanticRanker; r = MiniLMSemanticRanker()"
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings for all functions
- Keep functions focused and modular

## 📈 Performance Benchmarks

| Metric | Target | Achieved | Status |
|--------|--------|----------|---------|
| Processing Time | ≤60s | ~24s | ✅ |
| Model Size | ≤1GB | ~350MB | ✅ |
| CPU Only | Required | Yes | ✅ |
| No Internet | Required | Yes | ✅ |
| Document Count | 3-5 | 5 | ✅ |



## 📞 Support

For technical questions or issues:
1. Check the troubleshooting section above
2. Review `explanation.md` for detailed technical information
3. Open an issue in the repository

## 🙏 Acknowledgments

- Hugging Face for transformer models
- spaCy for NLP capabilities
- PyMuPDF for PDF processing
- The open-source ML community
