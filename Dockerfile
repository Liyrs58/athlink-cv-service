FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by opencv
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Create directories that the app needs at runtime
RUN mkdir -p /app/temp /app/memory/matches /app/static

CMD ["sh", "-c", "python startup_check.py && uvicorn main:app --host 0.0.0.0 --port ${PORT:-8005}"]
