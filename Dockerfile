FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies (lightweight — no torch, saves 2GB RAM)
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copy application
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Create data directory
RUN mkdir -p /app/backend/data

# Expose port
EXPOSE 8000

# Railway sets PORT env var
ENV PORT=8000

# Start server (run from backend/ so bare imports work)
CMD cd backend && python -m uvicorn main:app --host 0.0.0.0 --port ${PORT}
