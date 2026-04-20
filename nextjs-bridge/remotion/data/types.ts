// Shape matches services/export_service.build_export() output.

export type Vec2 = { x: number; y: number };

export type PlayerSample = {
  trackId: number;
  teamId: number;
  pitchX: number | null;
  pitchY: number | null;
  heading?: number;
  speed_kmh?: number;
  vector?: [number, number];
};

export type BallSample = {
  frameIndex: number;
  pitchX: number;
  pitchY: number;
  // Metres above the pitch plane. Non-zero only on frames where the ball
  // physics service detected a flight arc; 0 during ground rolls / no-cal.
  pitchZ?: number;
  confidence: number;
  reliable: string;
};

export type RawFrame = {
  frameIndex: number;
  timestampSeconds: number;
  players: PlayerSample[];
};

export type RawExport = {
  jobId: string;
  videoMeta: {
    width: number;
    height: number;
    fps: number;
    durationSeconds: number;
    frameCount: number;
  };
  teams: {
    team0: { color: string; playerCount: number };
    team1: { color: string; playerCount: number };
  };
  frames: RawFrame[];
  ball: BallSample[];
};

// Valid mocap states from backend
export type MocapLabel = "idle" | "run" | "sprint" | "dribble" | "kick";

// Normalized, render-ready per-track trajectory.
export type TrackTrajectory = {
  trackId: number;
  teamId: number;
  // samples indexed by frameIndex (NaN for missing frames)
  x: Float32Array;
  y: Float32Array;
  heading: Float32Array; // Degrees (0-360)
  speedKmh: Float32Array; // km/h from backend (NaN if unavailable)
  vectorX: Float32Array; // movement vector X component
  vectorY: Float32Array; // movement vector Y component
  mocapState: MocapLabel[]; // per-frame animation state from backend
};

export type BallTrajectory = {
  x: Float32Array; // NaN where unknown
  y: Float32Array;
  z: Float32Array; // metres above pitch, 0 when grounded or unknown
};

export type ReplayTable = {
  jobId: string;
  fps: number;
  frameCount: number;
  durationSeconds: number;
  team0Color: string;
  team1Color: string;
  tracks: TrackTrajectory[];
  ball: BallTrajectory;
  /** Per-frame timestamp in milliseconds from backend JSON (if available). */
  timestampMs: Float64Array | null;
};
