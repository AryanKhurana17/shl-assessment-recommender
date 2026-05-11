FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Render assigns a dynamic $PORT at runtime. 
# GOOGLE_API_KEY should be set in Render Environment variables.
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT