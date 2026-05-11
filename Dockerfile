FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (needed for some Python libraries if applicable)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# PRE-DOWNLOAD THE MODEL: This prevents the app from timing out on Render
# while trying to download the 80MB+ model during the first boot.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the rest of the application code
COPY . .

# Pre-build the FAISS index at build time
# (Ensure your scripts/build_index.py handles paths correctly)
RUN python -m scripts.build_index

# Render assigns a dynamic $PORT at runtime. 
# We use the shell form of CMD to ensure $PORT is correctly expanded.
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT