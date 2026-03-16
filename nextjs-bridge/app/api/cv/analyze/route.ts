import { cvFetch } from "@/lib/cv-client";
import { NextRequest } from "next/server";

export const revalidate = 0;

export async function POST(req: NextRequest) {
  const body = await req.json();

  // Submit tracking + team assignment job
  const trackRes = await cvFetch(
    "/api/v1/track/players-with-teams",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
    300_000
  );
  const trackData = await trackRes.json();

  if (!trackRes.ok) {
    return Response.json(trackData, { status: trackRes.status });
  }

  // If runPitchMap is requested, also submit pitch mapping
  if (body.runPitchMap) {
    const pitchRes = await cvFetch(
      "/api/v1/pitch/map",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          jobId: body.jobId,
          videoPath: body.videoPath,
        }),
      },
      300_000
    );
    const pitchData = await pitchRes.json();

    return Response.json(
      {
        trackingJobId: trackData.jobId,
        pitchJobId: pitchData.jobId ?? null,
      },
      { status: trackRes.status }
    );
  }

  return Response.json(trackData, { status: trackRes.status });
}
