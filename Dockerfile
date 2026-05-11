FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (includes pre-built FAISS index in data/faiss_index/)
COPY . .

# Render assigns a dynamic $PORT at runtime
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT