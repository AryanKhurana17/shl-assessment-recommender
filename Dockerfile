FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Pre-build the FAISS index at build time
RUN python -m scripts.build_index

EXPOSE 8000

# Use uvicorn with appropriate settings for production
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
