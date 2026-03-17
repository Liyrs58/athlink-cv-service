import { cvFetch } from "@/lib/cv-client";
import { NextRequest } from "next/server";

export const revalidate = 0;

export async function GET(
  req: NextRequest,
  { params }: { params: { jobId: string } }
) {
  const url = new URL(req.url);
  const query = url.searchParams.toString();
  const path = `/api/v1/set-pieces/${params.jobId}${query ? `?${query}` : ""}`;
  const res = await cvFetch(path);
  const data = await res.json();
  return Response.json(data, { status: res.status });
}
