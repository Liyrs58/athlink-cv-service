import { cvFetch } from "@/lib/cv-client";
import { NextRequest } from "next/server";

export const revalidate = 0;

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ jobId: string }> }
) {
  const { jobId } = await params;
  const trackId = req.nextUrl.searchParams.get("trackId");
  const path = trackId
    ? `/api/v1/stats/${jobId}/player/${trackId}`
    : `/api/v1/stats/${jobId}`;
  const res = await cvFetch(path);
  const data = await res.json();
  return Response.json(data, { status: res.status });
}
