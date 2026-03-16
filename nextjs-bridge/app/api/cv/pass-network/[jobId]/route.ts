import { cvFetch } from "@/lib/cv-client";
import { NextRequest } from "next/server";

export const revalidate = 0;

export async function GET(
  _req: NextRequest,
  { params }: { params: { jobId: string } }
) {
  const res = await cvFetch(`/api/v1/pass-network/${params.jobId}`);
  const data = await res.json();
  return Response.json(data, { status: res.status });
}
