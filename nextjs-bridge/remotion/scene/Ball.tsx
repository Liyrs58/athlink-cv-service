import * as THREE from "three";
import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { Trail } from "@react-three/drei";
import type { ReplayTable } from "../data/types";
import { BALL_RADIUS, pitchToWorld } from "./constants";

type Props = {
  table: ReplayTable;
  frame: number;
  simulatedTarget?: { x: number; z: number } | null;
};

/**
 * Ball position is read DIRECTLY from ball.x[frame], ball.y[frame].
 * No spline interpolation — frame = video frame_id, 1:1.
 *
 * Fallback when ball data is NaN at this frame:
 *   1. Search backward up to 8 frames for last known position.
 *   2. Search forward up to 8 frames for next known position.
 *   3. Lerp between them.
 *   4. If still nothing, place ball at nearest player to last known ball pos.
 */
export const Ball: React.FC<Props> = ({ table, frame, simulatedTarget }) => {
  const ref = useRef<THREE.Mesh>(null!);
  const lastPos = useRef(new THREE.Vector3(0, BALL_RADIUS, 0));
  const simulationStartTime = useRef<number | null>(null);
  const simulationStartPos = useRef(new THREE.Vector3());

  useFrame((state, dt) => {
    if (!ref.current) return;

    // ── Sandbox simulation mode ──
    if (simulatedTarget) {
      if (simulationStartTime.current === null) {
        simulationStartTime.current = state.clock.getElapsedTime();
        simulationStartPos.current.copy(ref.current.position);
      }
      const elapsed = state.clock.getElapsedTime() - simulationStartTime.current;
      const t = Math.min(elapsed / 1.5, 1);
      const tx = THREE.MathUtils.lerp(simulationStartPos.current.x, simulatedTarget.x, t);
      const tz = THREE.MathUtils.lerp(simulationStartPos.current.z, simulatedTarget.z, t);
      const ty = BALL_RADIUS + 4 * 3 * t * (1 - t);
      ref.current.position.set(tx, ty, tz);
      rollBall(ref.current, lastPos.current, dt);
      lastPos.current.copy(ref.current.position);
      return;
    }
    simulationStartTime.current = null;

    // ── Direct index into ball arrays ──
    const bx = table.ball.x[frame];
    const by = table.ball.y[frame];
    const bz = table.ball.z[frame];
    const hasData = bx !== undefined && by !== undefined && !Number.isNaN(bx) && !Number.isNaN(by);

    // KILL-SWITCH: ball at (0,0) = ghost ball → hide entirely
    if (hasData && bx === 0 && by === 0) {
      ref.current.visible = false;
      return;
    }

    if (hasData) {
      const [wx, wz] = pitchToWorld(bx, by);
      const h = (bz !== undefined && !Number.isNaN(bz) && bz > 0.3) ? bz : 0;
      ref.current.position.set(wx, BALL_RADIUS + h, wz);
      ref.current.visible = true;
    } else {
      // Fallback: lerp between nearest known frames
      const resolved = resolveNaN(table, frame);
      if (resolved) {
        ref.current.position.set(resolved.x, BALL_RADIUS + resolved.h, resolved.z);
        ref.current.visible = true;
      } else {
        // Last resort: place at nearest player to last known ball position
        const playerPos = nearestPlayerPos(table, frame, lastPos.current);
        if (playerPos) {
          ref.current.position.set(playerPos.x, BALL_RADIUS, playerPos.z);
          ref.current.visible = true;
        } else {
          ref.current.visible = false;
        }
      }
    }

    rollBall(ref.current, lastPos.current, dt);
    lastPos.current.copy(ref.current.position);
  });

  return (
    <Trail width={0.6} length={6} color="#ffffff" attenuation={(t) => t * t * t}>
      <mesh ref={ref} castShadow>
        <sphereGeometry args={[BALL_RADIUS, 32, 32]} />
        <meshStandardMaterial color="white" roughness={0.3} metalness={0.1} />
      </mesh>
    </Trail>
  );
};

/** Roll the ball mesh based on distance traveled. */
function rollBall(mesh: THREE.Mesh, lastPos: THREE.Vector3, dt: number) {
  if (dt <= 0) return;
  const dist = mesh.position.distanceTo(lastPos);
  if (dist > 0.001) {
    const axis = new THREE.Vector3()
      .copy(mesh.position)
      .sub(lastPos)
      .cross(new THREE.Vector3(0, 1, 0))
      .normalize();
    if (axis.lengthSq() > 0.0001) {
      mesh.rotateOnWorldAxis(axis, dist / BALL_RADIUS);
    }
  }
}

/** Search backward/forward up to 8 frames for valid ball data, lerp between. */
function resolveNaN(
  table: ReplayTable,
  frame: number
): { x: number; z: number; h: number } | null {
  const maxSearch = 8;
  const len = table.ball.x.length;

  // Find last known before
  let prevI = -1;
  for (let k = 1; k <= maxSearch; k++) {
    const i = frame - k;
    if (i < 0) break;
    if (!Number.isNaN(table.ball.x[i]) && !Number.isNaN(table.ball.y[i])) {
      prevI = i;
      break;
    }
  }

  // Find next known after
  let nextI = -1;
  for (let k = 1; k <= maxSearch; k++) {
    const i = frame + k;
    if (i >= len) break;
    if (!Number.isNaN(table.ball.x[i]) && !Number.isNaN(table.ball.y[i])) {
      nextI = i;
      break;
    }
  }

  if (prevI < 0 && nextI < 0) return null;

  // Only one side available — use it directly
  if (prevI < 0 && nextI >= 0) {
    const [wx, wz] = pitchToWorld(table.ball.x[nextI], table.ball.y[nextI]);
    const bz = table.ball.z[nextI];
    return { x: wx, z: wz, h: (!Number.isNaN(bz) && bz > 0.3) ? bz : 0 };
  }
  if (nextI < 0 && prevI >= 0) {
    const [wx, wz] = pitchToWorld(table.ball.x[prevI], table.ball.y[prevI]);
    const bz = table.ball.z[prevI];
    return { x: wx, z: wz, h: (!Number.isNaN(bz) && bz > 0.3) ? bz : 0 };
  }

  // Both sides — lerp
  const t = (frame - prevI) / (nextI - prevI);
  const px = table.ball.x[prevI] + (table.ball.x[nextI] - table.ball.x[prevI]) * t;
  const py = table.ball.y[prevI] + (table.ball.y[nextI] - table.ball.y[prevI]) * t;
  const [wx, wz] = pitchToWorld(px, py);

  const pz = table.ball.z[prevI];
  const nz = table.ball.z[nextI];
  const validPz = !Number.isNaN(pz) ? pz : 0;
  const validNz = !Number.isNaN(nz) ? nz : 0;
  const h = validPz + (validNz - validPz) * t;

  return { x: wx, z: wz, h: h > 0.3 ? h : 0 };
}

/** Find the nearest player to lastBallPos at the given frame. */
function nearestPlayerPos(
  table: ReplayTable,
  frame: number,
  lastBallPos: THREE.Vector3
): { x: number; z: number } | null {
  let bestDist = Infinity;
  let bestWx = 0, bestWz = 0;
  let found = false;

  for (const trk of table.tracks) {
    const px = trk.x[frame];
    const py = trk.y[frame];
    if (px === undefined || py === undefined || Number.isNaN(px) || Number.isNaN(py)) continue;
    const [wx, wz] = pitchToWorld(px, py);
    const d = Math.hypot(wx - lastBallPos.x, wz - lastBallPos.z);
    if (d < bestDist) {
      bestDist = d;
      bestWx = wx;
      bestWz = wz;
      found = true;
    }
  }

  return found ? { x: bestWx, z: bestWz } : null;
}
