import { cvFetch } from "@/lib/cv-client";
import { NextRequest } from "next/server";

export const revalidate = 0;

export async function POST(req: NextRequest) {
  const body = await req.json();

  const res = await cvFetch(
    "/api/v1/spotlight/render",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
    300_000
  );
  const data = await res.json();
  return Response.json(data, { status: res.status });
}
