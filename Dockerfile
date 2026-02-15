# Use stable Python base
FROM python:3.10-slim

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# Copy only requirements first (critical for caching)
# -------------------------
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies (no cache reduces image size)
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader stopwords

# -------------------------
# Now copy the full project
# -------------------------
COPY . .

# Expose port
EXPOSE 7860

# Production server
CMD ["gunicorn", "--chdir", "src", "app:app", "--bind", "0.0.0.0:7860"]
