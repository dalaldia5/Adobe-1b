# Use a specific Python version with CPU support
FROM python:3.9.16-slim-bullseye

# Set the working directory inside the container
WORKDIR /app

# Set environment variables to cache models inside the container for offline use
ENV HF_HOME=/app/huggingface_cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/huggingface_cache
ENV HF_HUB_DISABLE_PROGRESS_BARS=1
ENV PYTHONUNBUFFERED=1
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV TRANSFORMERS_OFFLINE=0
ENV HF_DATASETS_OFFLINE=0

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt .

# Install dependencies with optimized versions for CPU-only BART usage
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir numpy>=1.21.0 && \
    pip install --no-cache-dir scipy>=1.7.0 && \
    pip install --no-cache-dir torch>=1.9.0 --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir transformers>=4.20.0 && \
    pip install --no-cache-dir sentence-transformers>=2.2.0 && \
    pip install --no-cache-dir PyMuPDF>=1.20.0 && \
    pip install --no-cache-dir scikit-learn>=1.0.0

# Download and cache models for offline use (BART + sentence transformers)
RUN python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
    print('Downloading BART model...'); \
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-cnn'); \
    model = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-large-cnn'); \
    print('Downloading sentence transformer...'); \
    from sentence_transformers import SentenceTransformer; \
    st_model = SentenceTransformer('all-MiniLM-L6-v2'); \
    print('Models cached successfully');"

# Set environment variables back to offline mode after downloading
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

# Copy the rest of the application code into the container
COPY approach_explanation_new.md .
COPY README.md .
COPY solution /app/solution

# Define the entry point for the container. This will run the main script.
CMD ["python", "-m", "solution.main"]