import { cvFetch } from "@/lib/cv-client";
import { NextRequest } from "next/server";

export const revalidate = 0;

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ jobId: string; team: string }> }
) {
  const { jobId, team } = await params;
  const path = `/api/v1/reports/${jobId}/team/${team}`;
  const res = await cvFetch(path);

  if (!res.ok) {
    const data = await res.json().catch(() => ({ error: "Failed to generate report" }));
    return Response.json(data, { status: res.status });
  }

  const bytes = await res.arrayBuffer();
  return new Response(bytes, {
    status: 200,
    headers: {
      "Content-Type": "application/pdf",
      "Content-Disposition": `attachment; filename="team_${team}_report.pdf"`,
    },
  });
}
