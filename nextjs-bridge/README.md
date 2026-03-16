# Next.js CV Service Bridge

API route files that proxy requests from the Next.js app to the Python CV service.

## Setup

Copy these files into your Next.js App Router project:

```bash
# From the Next.js project root:
cp -r /path/to/nextjs-bridge/lib/ ./lib/
cp -r /path/to/nextjs-bridge/app/api/cv/ ./app/api/cv/
```

## Environment Variable

Add to your `.env.local`:

```
CV_SERVICE_URL=http://127.0.0.1:8001
```

In production (e.g. Vercel), set `CV_SERVICE_URL` to the deployed Python service URL.

## Routes

| Next.js Route | Method | CV Service Endpoint | Timeout |
|---|---|---|---|
| `/api/cv/analyze` | POST | `/api/v1/track/players-with-teams` | 300s |
| `/api/cv/status/[jobId]` | GET | `/api/v1/jobs/status/{jobId}` | 30s |
| `/api/cv/stats/[jobId]` | GET | `/api/v1/stats/{jobId}` | 30s |
| `/api/cv/analytics/[jobId]` | GET | `/api/v1/analytics/{jobId}` | 30s |
| `/api/cv/render` | POST | `/api/v1/render/{jobId}` | 300s |
| `/api/cv/spotlight` | POST | `/api/v1/spotlight/render` | 300s |
| `/api/cv/highlights/[jobId]` | GET | `/api/v1/highlight/detect` | 30s |
| `/api/cv/pass-network/[jobId]` | GET | `/api/v1/pass-network/{jobId}` | 30s |
| `/api/cv/xg/[jobId]` | GET | `/api/v1/xg/{jobId}` | 30s |
| `/api/cv/heatmap/[jobId]` | GET | `/api/v1/heatmap/{jobId}` | 30s |
| `/api/cv/pressing/[jobId]` | GET | `/api/v1/pressing/{jobId}` | 30s |
| `/api/cv/formation/[jobId]` | GET | `/api/v1/formation/{jobId}` | 30s |
| `/api/cv/storage/upload/[jobId]` | POST | `/api/v1/storage/upload/{jobId}` | 30s |
