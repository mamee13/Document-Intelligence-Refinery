# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install system dependencies for Docling (OCR, Layout, and Image Processing)
RUN apt-get update && apt-get install -y \
    build-essential \
    libmagic-dev \
    tesseract-ocr \
    libtesseract-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Install uv for fast package management
RUN pip install uv

# Copy the requirements and lock file
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
# --no-install-project is used since we copy code later
RUN uv sync --frozen --no-install-project

# Copy the rest of the application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

# Create a directory for all potential outputs
RUN mkdir -p .refinery/profiles .refinery/extracted .refinery/pageindex .refinery/vectorstore

# Using src/main.py as the entry point for the CLI refinery
ENTRYPOINT ["python", "-m", "src.main"]
CMD ["extract"]
