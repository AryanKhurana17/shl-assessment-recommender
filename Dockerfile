FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Prevent OpenMP runtime errors
ENV KMP_DUPLICATE_LIB_OK=TRUE

# Render assigns a dynamic $PORT at runtime. 
# We use uvicorn directly for simplicity in the CMD.
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT