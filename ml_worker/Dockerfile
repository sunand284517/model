FROM python:3.10-slim

WORKDIR /app

# Install system dependencies needed for PyTorch and Celery
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the python files
COPY . .

# Run the Celery worker
CMD ["celery", "-A", "worker.app", "worker", "--loglevel=info"]