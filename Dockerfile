FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Create data directory
RUN mkdir -p /app/backend/data

# Expose port
EXPOSE 8000

# Railway sets PORT env var
ENV PORT=8000

# Start server
CMD python -m uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}
