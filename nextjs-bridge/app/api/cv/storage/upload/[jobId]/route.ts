import { cvFetch } from "@/lib/cv-client";
import { NextRequest } from "next/server";

export const revalidate = 0;

export async function POST(
  _req: NextRequest,
  { params }: { params: { jobId: string } }
) {
  const res = await cvFetch(
    `/api/v1/storage/upload/${params.jobId}`,
    { method: "POST" }
  );
  const data = await res.json();
  return Response.json(data, { status: res.status });
}
