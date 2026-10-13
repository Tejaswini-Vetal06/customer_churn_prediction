# Production Dockerfile for Customer Churn FastAPI & Streamlit Services
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and artifacts
COPY src/ /app/src/
COPY api/ /app/api/
COPY app/ /app/app/
COPY artifacts/ /app/artifacts/
COPY data/ /app/data/

# Default exposes FastAPI on 8000
EXPOSE 8000
EXPOSE 8501

# Default start command launches FastAPI
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

