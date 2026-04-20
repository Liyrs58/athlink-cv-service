import type { ReplayTable, TrackTrajectory, BallTrajectory } from "./types";
import { PITCH_LENGTH, PITCH_WIDTH } from "../scene/constants";

/**
 * Generate synthetic ReplayTable data for the Remotion preview scaffold.
 * Produces 22 players (11 per team) with gentle movement animations
 * and a ball that moves across the pitch — enough to verify lighting,
 * camera, player rendering, and ball rendering without a real export.
 */
export function generateDemoTable(
  fps = 25,
  durationSeconds = 10
): ReplayTable {
  const frameCount = fps * durationSeconds;

  // ── players: 11 per team in a rough formation ──────────────────
  const team0Positions = formation442(35, PITCH_WIDTH / 2); // left half
  const team1Positions = formation442(70, PITCH_WIDTH / 2); // right half

  const tracks: TrackTrajectory[] = [];

  const addTeam = (
    offsets: [number, number][],
    teamId: number,
    startId: number
  ) => {
    offsets.forEach(([bx, by], i) => {
      const x = new Float32Array(frameCount);
      const y = new Float32Array(frameCount);
      for (let f = 0; f < frameCount; f++) {
        const t = f / frameCount;
        // gentle lateral drift + small oscillation
        x[f] = bx + Math.sin(t * Math.PI * 2 + i * 0.7) * 3;
        y[f] = by + Math.cos(t * Math.PI * 2 + i * 1.1) * 2;
      }
      const heading = new Float32Array(frameCount).fill(NaN);
      const speedKmh = new Float32Array(frameCount).fill(NaN);
      const vectorX = new Float32Array(frameCount).fill(NaN);
      const vectorY = new Float32Array(frameCount).fill(NaN);
      const mocapState = new Array(frameCount).fill("idle") as import("./types").MocapLabel[];
      tracks.push({ trackId: startId + i, teamId, x, y, heading, speedKmh, vectorX, vectorY, mocapState });
    });
  };

  addTeam(team0Positions, 0, 1);
  addTeam(team1Positions, 1, 12);

  // ── ball: sweeps from left half to right with a lob arc ────────
  const ballX = new Float32Array(frameCount);
  const ballY = new Float32Array(frameCount);
  const ballZ = new Float32Array(frameCount);
  for (let f = 0; f < frameCount; f++) {
    const t = f / frameCount;
    ballX[f] = 20 + t * 65; // travel across most of the pitch
    ballY[f] = PITCH_WIDTH / 2 + Math.sin(t * Math.PI * 3) * 12;
    // parabolic lob in the middle third
    const lobPhase = Math.max(0, Math.sin(t * Math.PI));
    ballZ[f] = lobPhase * 4; // up to 4 m height
  }
  const ball: BallTrajectory = { x: ballX, y: ballY, z: ballZ };

  const timestampMs = new Float64Array(frameCount);
  for (let i = 0; i < frameCount; i++) timestampMs[i] = (i / fps) * 1000;

  return {
    jobId: "demo",
    fps,
    frameCount,
    durationSeconds,
    team0Color: "#0064FF",
    team1Color: "#FF3200",
    tracks,
    ball,
    timestampMs,
  };
}

/** 4-4-2 formation positions (pitch coordinates: 0..105 x 0..68) */
function formation442(
  centreX: number,
  centreY: number
): [number, number][] {
  const dx = centreX < PITCH_LENGTH / 2 ? -1 : 1; // direction
  return [
    // GK
    [centreX - dx * 30, centreY],
    // Back 4
    [centreX - dx * 18, centreY - 18],
    [centreX - dx * 18, centreY - 6],
    [centreX - dx * 18, centreY + 6],
    [centreX - dx * 18, centreY + 18],
    // Mid 4
    [centreX - dx * 6, centreY - 18],
    [centreX - dx * 6, centreY - 6],
    [centreX - dx * 6, centreY + 6],
    [centreX - dx * 6, centreY + 18],
    // Strikers 2
    [centreX + dx * 8, centreY - 8],
    [centreX + dx * 8, centreY + 8],
  ];
}
