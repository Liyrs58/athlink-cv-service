import type {
  RawExport,
  ReplayTable,
  TrackTrajectory,
  BallTrajectory,
} from "./types";

/**
 * Fetch /api/v1/export/{jobId} and normalize into dense per-track Float32 arrays
 * indexed by frameIndex.
 *
 * NaN marks frames where a given track (or the ball) has no observation.
 * Interpolation/carry-forward is the renderer's responsibility — the loader
 * preserves ground truth.
 */
/**
 * Specialized loader for FC26 live simulation data.
 * Fetches from the /public/data directory.
 */
export async function loadReplayData(): Promise<ReplayTable> {
  const response = await fetch("/data/fc26_live_data.json");
  if (!response.ok) {
    throw new Error(`Failed to load simulation data: ${response.status} ${response.statusText}`);
  }
  const raw = (await response.json()) as RawExport;
  return normalize(raw);
}

export async function loadExport(url: string): Promise<ReplayTable> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`export fetch failed: ${res.status} ${res.statusText}`);
  }
  const raw = (await res.json()) as RawExport;
  return normalize(raw);
}

export function normalize(raw: RawExport): ReplayTable {
  const frameCount = raw.videoMeta.frameCount || raw.frames.length;
  const fps = raw.videoMeta.fps || 30;

  // Discover all track IDs and teams
  const trackMap = new Map<number, TrackTrajectory>();

  for (const f of raw.frames) {
    for (const p of f.players) {
      if (!trackMap.has(p.trackId)) {
        const x = new Float32Array(frameCount);
        const y = new Float32Array(frameCount);
        const heading = new Float32Array(frameCount);
        x.fill(NaN);
        y.fill(NaN);
        heading.fill(NaN);
        trackMap.set(p.trackId, {
          trackId: p.trackId,
          teamId: p.teamId,
          x,
          y,
          heading,
        });
      }
      const trk = trackMap.get(p.trackId)!;
      if (p.teamId !== -1) trk.teamId = p.teamId; // take first non-unknown
      const fi = f.frameIndex;
      if (fi >= 0 && fi < frameCount && p.pitchX !== null && p.pitchY !== null) {
        trk.x[fi] = p.pitchX;
        trk.y[fi] = p.pitchY;
        trk.heading[fi] = p.heading ?? NaN;
      }
    }
  }

  // Ball
  const ballX = new Float32Array(frameCount);
  const ballY = new Float32Array(frameCount);
  const ballZ = new Float32Array(frameCount); // defaults to 0 (grounded)
  ballX.fill(NaN);
  ballY.fill(NaN);
  for (const b of raw.ball ?? []) {
    if (b.frameIndex >= 0 && b.frameIndex < frameCount) {
      ballX[b.frameIndex] = b.pitchX;
      ballY[b.frameIndex] = b.pitchY;
      ballZ[b.frameIndex] = b.pitchZ ?? 0;
    }
  }
  const ball: BallTrajectory = { x: ballX, y: ballY, z: ballZ };

  const tracks = Array.from(trackMap.values()).sort(
    (a, b) => a.trackId - b.trackId
  );

  return {
    jobId: raw.jobId,
    fps,
    frameCount,
    durationSeconds: raw.videoMeta.durationSeconds || frameCount / fps,
    team0Color: raw.teams?.team0?.color ?? "#0064FF",
    team1Color: raw.teams?.team1?.color ?? "#FF3200",
    tracks,
    ball,
  };
}

/**
 * Sample a Float32 trajectory at a fractional frame index.
 * Returns NaN if surrounded by NaN beyond a small carry-forward window.
 */
export function sampleAt(
  arr: Float32Array,
  frame: number,
  carryForwardFrames = 8
): number {
  if (frame < 0 || frame >= arr.length) return NaN;
  const i0 = Math.floor(frame);
  const i1 = Math.min(i0 + 1, arr.length - 1);
  const t = frame - i0;
  const v0 = arr[i0];
  const v1 = arr[i1];

  if (!Number.isNaN(v0) && !Number.isNaN(v1)) {
    return v0 * (1 - t) + v1 * t;
  }
  if (!Number.isNaN(v0)) return v0;
  if (!Number.isNaN(v1)) return v1;

  // Carry forward from the nearest prior known value
  for (let k = 1; k <= carryForwardFrames && i0 - k >= 0; k++) {
    const v = arr[i0 - k];
    if (!Number.isNaN(v)) return v;
  }
  return NaN;
}

/**
 * Smooth spline sampler — cubic Hermite with Catmull-Rom tangents.
 *
 * Handles the sparse-NaN reality of tracking data:
 *  - finds nearest valid samples on each side of `frame`
 *  - computes tangents from a second neighbor on each side (Catmull-Rom)
 *  - degrades to linear at boundaries, hold-last across big gaps
 *
 * gapThreshold: if nearest valid samples straddling `frame` are more than N
 * frames apart, don't spline across the gap — just hold the last value. Keeps
 * a spline from drawing a graceful arc across a 2-second occlusion.
 */
export function sampleSpline(
  arr: Float32Array,
  frame: number,
  gapThreshold = 15,
  carryForwardFrames = 8
): number {
  if (frame < 0 || frame >= arr.length) return NaN;
  const i0 = Math.floor(frame);

  // Locate p1 = nearest valid at-or-before frame
  let p1i = i0;
  if (Number.isNaN(arr[p1i])) {
    let k = 1;
    while (
      k <= carryForwardFrames &&
      p1i - k >= 0 &&
      Number.isNaN(arr[p1i - k])
    )
      k++;
    if (p1i - k < 0 || k > carryForwardFrames) return NaN;
    p1i -= k;
  }
  // p2 = nearest valid strictly after p1
  let p2i = p1i + 1;
  while (p2i < arr.length && Number.isNaN(arr[p2i])) p2i++;
  if (p2i >= arr.length) return arr[p1i]; // past end → hold last
  if (p2i - p1i > gapThreshold) return arr[p1i]; // gap too wide → hold
  if (frame <= p1i) return arr[p1i]; // before first known on this side

  // Neighbors for tangents
  let p0i = p1i - 1;
  while (p0i >= 0 && Number.isNaN(arr[p0i])) p0i--;
  let p3i = p2i + 1;
  while (p3i < arr.length && Number.isNaN(arr[p3i])) p3i++;

  const v1 = arr[p1i];
  const v2 = arr[p2i];
  const dt = p2i - p1i;
  const u = (frame - p1i) / dt;

  // Catmull-Rom tangents (one-sided at boundaries)
  const m1 =
    p0i >= 0 ? (v2 - arr[p0i]) / (p2i - p0i) : (v2 - v1) / dt;
  const m2 =
    p3i < arr.length ? (arr[p3i] - v1) / (p3i - p1i) : (v2 - v1) / dt;

  const u2 = u * u;
  const u3 = u2 * u;
  const h00 = 2 * u3 - 3 * u2 + 1;
  const h10 = u3 - 2 * u2 + u;
  const h01 = -2 * u3 + 3 * u2;
  const h11 = u3 - u2;

  return h00 * v1 + h10 * dt * m1 + h01 * v2 + h11 * dt * m2;
}

/**
 * Specialized angle sampler (degrees) with short-est path interpolation.
 * Handles 0/360 wrap-around.
 */
export function sampleAngle(
  arr: Float32Array,
  frame: number,
  carryForwardFrames = 8
): number {
  if (frame < 0 || frame >= arr.length) return NaN;
  const i0 = Math.floor(frame);
  const i1 = Math.min(i0 + 1, arr.length - 1);
  const t = frame - i0;
  const v0 = arr[i0];
  const v1 = arr[i1];

  if (!Number.isNaN(v0) && !Number.isNaN(v1)) {
    // Shortest path interpolation (0-360)
    let diff = v1 - v0;
    while (diff < -180) diff += 360;
    while (diff > 180) diff -= 360;
    return (v0 + diff * t + 360) % 360;
  }

  if (!Number.isNaN(v0)) return v0;
  if (!Number.isNaN(v1)) return v1;

  for (let k = 1; k <= carryForwardFrames && i0 - k >= 0; k++) {
    const v = arr[i0 - k];
    if (!Number.isNaN(v)) return v;
  }
  return NaN;
}

export function sampleSplineAngle(
  arr: Float32Array,
  frame: number,
  gapThreshold = 15,
  carryForwardFrames = 8
): number {
  // Simple angle linear fallback for now, as spline-ing across circle wrap is non-trivial
  // and linear "shortest path" is usually sufficient for headings.
  return sampleAngle(arr, frame, carryForwardFrames);
}
