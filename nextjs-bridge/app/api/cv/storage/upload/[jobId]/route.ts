import { cvFetch } from "@/lib/cv-client";
import { NextRequest } from "next/server";

export const revalidate = 0;

export async function POST(
  _req: NextRequest,
  { params }: { params: Promise<{ jobId: string }> }
) {
  const { jobId } = await params;
  const res = await cvFetch(
    `/api/v1/storage/upload/${jobId}`,
    { method: "POST" }
  );
  const data = await res.json();
  return Response.json(data, { status: res.status });
}
