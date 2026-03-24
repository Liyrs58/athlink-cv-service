import { cvFetch } from "@/lib/cv-client";
import { NextRequest } from "next/server";

export const revalidate = 0;

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ jobId: string }> }
) {
  const { jobId } = await params;
  const trackId = req.nextUrl.searchParams.get("trackId");
  const qs = trackId ? `?trackId=${trackId}` : "";
  const res = await cvFetch(`/api/v1/heatmap/${jobId}${qs}`);
  const data = await res.json();
  return Response.json(data, { status: res.status });
}
