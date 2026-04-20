import { useEffect, useState } from "react";
import {
  AbsoluteFill,
  continueRender,
  delayRender,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { ThreeCanvas } from "@remotion/three";
import { z } from "zod";
import { loadExport, loadReplayData } from "../data/loader";
import { generateDemoTable } from "../data/demo";
import type { ReplayTable } from "../data/types";
import { Stage } from "../scene/Stage";

export const pitchReplaySchema = z.object({
  jobId: z.string(),
  exportUrl: z.string().url().nullable(),
  videoFps: z.number().default(25),
});

type Props = z.infer<typeof pitchReplaySchema>;

export const PitchReplay: React.FC<Props> = ({ jobId, exportUrl, videoFps }) => {
  const frame = useCurrentFrame();
  const { width, height, fps, durationInFrames } = useVideoConfig();

  const [handle] = useState(() => delayRender(`load-export-${jobId}`));
  const [table, setTable] = useState<ReplayTable | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const isLive = jobId === "fc26_live_simulation";
    const targetUrl = exportUrl || "http://127.0.0.1:8001/api/v1/export/e2e_v2";

    const fetchPromise = isLive ? loadReplayData() : loadExport(targetUrl);

    fetchPromise
      .then((t: ReplayTable) => {
        setTable(t);
        if (isLive) console.log("🚀 FC26 Simulation Loaded:", t.jobId);
        continueRender(handle);
      })
      .catch((e: Error) => {
        console.warn("Real data fetch failed, falling back to synthetic", e.message);
        setTable(generateDemoTable(fps, durationInFrames / fps));
        continueRender(handle);
      });
  }, [exportUrl, handle, fps, durationInFrames, jobId]);

  if (error) {
    return (
      <AbsoluteFill style={{
        backgroundColor: "#0a0f0d", color: "#ff6b6b", fontFamily: "monospace",
        display: "flex", alignItems: "center", justifyContent: "center",
        fontSize: 32, padding: 64, textAlign: "center",
      }}>
        <div>
          <div>Athlink CV — Pitch Replay</div>
          <div style={{ fontSize: 20, marginTop: 16, opacity: 0.8 }}>job: {jobId}</div>
          <div style={{ fontSize: 20, marginTop: 32, color: "#ff9a9a" }}>{error}</div>
        </div>
      </AbsoluteFill>
    );
  }

  if (!table) {
    return (
      <AbsoluteFill style={{
        backgroundColor: "#0a0f0d", color: "#9fe8b8", fontFamily: "monospace",
        display: "flex", alignItems: "center", justifyContent: "center", fontSize: 40,
      }}>
        Loading {jobId}…
      </AbsoluteFill>
    );
  }

  return (
    <AbsoluteFill style={{ backgroundColor: "#000" }}>
      <ThreeCanvas
        width={width}
        height={height}
        shadows="soft"
        gl={{ antialias: true, toneMapping: 4 /* ACESFilmic */ }}
        dpr={1}
      >
        <Stage table={table} frame={frame} videoFps={videoFps} />
      </ThreeCanvas>
    </AbsoluteFill>
  );
};
