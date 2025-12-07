# Dockerfile for CLIP Prefix Captioning Model (HuggingFace Deployment)
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements-hf.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements-hf.txt

# Install CLIP from GitHub
RUN pip3 install git+https://github.com/openai/CLIP.git

# Copy application code
COPY inference_api_hf.py .

# Create necessary directories
RUN mkdir -p /app/hf_cache

# Download NLTK data
RUN python3 -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# Expose port for API
EXPOSE 8000

# Set entrypoint
CMD ["python3", "inference_api_hf.py"]

