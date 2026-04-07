FROM python:3.11-slim

WORKDIR /app

# System deps for opencv-python-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/temp /app/memory/matches /app/static /app/weights

# Pre-download football YOLO model at build time (non-fatal — falls back to yolov8s at runtime)
RUN python -c "from model_downloader import download_football_model; download_football_model()" || \
    echo "WARNING: Football model download failed — will use fallback model at runtime"

CMD ["python", "main.py"]
