# Dockerfile for EVE Price Prediction Engine
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code.
# No .env is baked into the image: environment variables (set by
# docker-compose) are the single source of truth for configuration. A baked-in
# .env.example would otherwise ship placeholder credentials and a
# MLFLOW_TRACKING_URI pointing at the container's own localhost.
COPY src/ ./src/
COPY scripts/ ./scripts/

# Create data directories and run as a non-root user
RUN mkdir -p data/raw data/processed data/models && \
    useradd --create-home --uid 10001 eve && \
    chown -R eve:eve /app
USER eve

# Expose API port
EXPOSE 8000

# Default command (can be overridden in docker-compose)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
