# AthLink CV Service

Football video analysis backend service built with FastAPI.

## Goal

Build a local computer-vision backend that will handle:
- Video ingestion
- Frame extraction
- Player detection
- Player tracking
- Team color separation
- Structured JSON output

## Project Structure

```
athlink-cv-service/
├── main.py              # FastAPI application entry point
├── requirements.txt     # Python dependencies
├── .env.example        # Environment variables template
├── routes/             # API route handlers
│   ├── __init__.py
│   ├── health.py       # Health check endpoint
│   └── analyze.py      # Video analysis endpoints
├── services/           # Business logic layer
│   ├── __init__.py
│   └── job_service.py  # Job management service
├── models/             # Pydantic models
│   ├── __init__.py
│   └── analysis.py     # Analysis request/response models
├── outputs/            # Analysis results output directory
└── temp/              # Temporary files directory
```

## Installation & Setup

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Server

```bash
uvicorn main:app --reload
```

The server will start on `http://localhost:8000`

### 4. Test Health Endpoint

```bash
curl http://localhost:8000/api/v1/health
```

Expected response:
```json
{
  "status": "ok",
  "service": "athlink-cv-service"
}
```

## API Endpoints

### Health Check
- **GET** `/api/v1/health` - Service health status

### Video Analysis
- **POST** `/api/v1/analyze` - Submit video for analysis

Request body:
```json
{
  "jobId": "job_123",
  "videoPath": "/path/to/video.mp4"
}
```

Response:
```json
{
  "jobId": "job_123",
  "status": "queued",
  "message": "video accepted for analysis",
  "video": {
    "path": "/path/to/video.mp4",
    "filename": "video.mp4",
    "extension": ".mp4",
    "sizeBytes": 123456,
    "width": 1920,
    "height": 1080,
    "fps": 25.0,
    "frameCount": 500,
    "durationSeconds": 20.0
  }
}
```

### Video Inspection
- **POST** `/api/v1/video/inspect` - Get video metadata without creating a job

Request body:
```json
{
  "jobId": "job_123",
  "videoPath": "/path/to/video.mp4"
}
```

Response:
```json
{
  "jobId": "job_123",
  "video": {
    "path": "/path/to/video.mp4",
    "filename": "video.mp4",
    "extension": ".mp4",
    "sizeBytes": 123456,
    "width": 1920,
    "height": 1080,
    "fps": 25.0,
    "frameCount": 500,
    "durationSeconds": 20.0
  }
}
```

### Player Detection
- **POST** `/api/v1/detect/players` - Run player detection on sampled video frames

Request body:
```json
{
  "jobId": "job_123",
  "videoPath": "/path/to/video.mp4",
  "sampleCount": 3
}
```

Response:
```json
{
  "jobId": "job_123",
  "sampleCount": 3,
  "frames": [
    {
      "frameIndex": 0,
      "timestampSeconds": 0.0,
      "imagePath": "temp/job_123/frames/frame_000000.jpg",
      "detections": [
        {
          "className": "person",
          "confidence": 0.91,
          "bbox": {
            "x": 120,
            "y": 80,
            "width": 32,
            "height": 90
          }
        }
      ]
    }
  ]
}
```

### Frame Sampling
- **POST** `/api/v1/video/sample-frames` - Extract and sample frames from video

Request body:
```json
{
  "jobId": "job_123",
  "videoPath": "/path/to/video.mp4",
  "sampleCount": 3
}
```

Response:
```json
{
  "jobId": "job_123",
  "sampleCount": 3,
  "frames": [
    {
      "frameIndex": 0,
      "timestampSeconds": 0.0,
      "imagePath": "temp/job_123/frames/frame_000000.jpg"
    },
    {
      "frameIndex": 250,
      "timestampSeconds": 10.0,
      "imagePath": "temp/job_123/frames/frame_000250.jpg"
    },
    {
      "frameIndex": 499,
      "timestampSeconds": 19.96,
      "imagePath": "temp/job_123/frames/frame_000499.jpg"
    }
  ]
}
```

## Development

The API documentation is available at `http://localhost:8000/docs`

## Future Implementation

- [ ] Video file upload handling
- [ ] Frame extraction service
- [ ] Player detection algorithms
- [ ] Player tracking implementation
- [ ] Team color classification
- [ ] WebSocket for real-time progress updates
- [ ] Job queue management (Redis/Celery)
- [ ] File storage management
- [ ] Error handling and logging
- [ ] Authentication and authorization
