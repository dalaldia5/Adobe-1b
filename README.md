# Persona-Driven Document Intelligence - Solution 1B

## Core Strategy

This solution is built on a three-pillar strategy for maximum performance and relevance:

1.  **Intelligent Chunking**: Instead of naive fixed-size splitting, we parse documents by paragraphs. This preserves the semantic integrity of the text, leading to higher-quality embeddings and more relevant search results.

2.  **Optimized Semantic Search**: We use the `all-MiniLM-L6-v2` model, which offers an exceptional balance of speed, a small memory footprint (~80MB), and high performance for semantic similarity tasks. This is the core of our relevance engine.

3.  **Refined Sub-Section Analysis**: For the top-ranked sections, we use a lightweight extractive summarizer (`bert-extractive-summarizer`) to distill the most critical sentences. This provides a granular and focused analysis that directly addresses the "Sub-Section Relevance" scoring criteria.

## Tech Stack

- **Semantic Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Summarization**: `bert-extractive-summarizer` with `paraphrase-MiniLM-L6-v2`
- **PDF Parsing**: `PyMuPDF`
- **Core Libraries**: `PyTorch`, `scikit-learn`
- **Containerization**: `Docker`

## How to Build and Run

### 1. Build the Docker Image

From the root directory of this project, run the build command. This will install all dependencies and pre-cache the required models for offline execution, as per the hackathon rules.

```bash
docker build -t solution-1b.
```
