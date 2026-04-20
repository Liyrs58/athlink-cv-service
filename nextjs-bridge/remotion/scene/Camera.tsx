import * as THREE from "three";
import { useEffect, useRef } from "react";
import { useThree, useFrame } from "@react-three/fiber";
import type { ReplayTable } from "../data/types";
import { sampleAt, sampleSpline, sampleSplineAngle } from "../data/loader";
import { PITCH_LENGTH, PITCH_WIDTH, pitchToWorld } from "./constants";

type Props = { 
  table: ReplayTable; 
  frame: number; 
  povPlayerId?: string | number | null; // Track ID to follow in POV mode
};

const HEAD_H = 1.72;
const POV_BEHIND = 0.15;
const BROADCAST_H = 22;
const BROADCAST_D = 30;
const MIN_HOLD = 60; // frames — minimum shot duration before cut
const BALL_LEAD = 5; // metres ahead of ball velocity vector

export const Camera: React.FC<Props> = ({ table, frame, povPlayerId }) => {
  const camera = useThree((s) => s.camera) as THREE.PerspectiveCamera;

  const sm = useRef({
    pos: new THREE.Vector3(0, BROADCAST_H, BROADCAST_D),
    look: new THREE.Vector3(),
    fov: 45,
    mode: "broadcast" as "broadcast" | "pov",
    lastSwitch: 0,
    // Broadcast smoothing
    targetX: 0,
    targetZ: 0,
    ballFollowW: 0,
    lastBallX: NaN,
    lastBallZ: NaN,
  });

  useEffect(() => {
    camera.near = 1;
    camera.far = 400;
    camera.fov = 45;
    camera.updateProjectionMatrix();
  }, [camera]);

  useFrame(() => {
    const s = sm.current;

    // ── Ball state ──
    const bx = sampleSpline(table.ball.x, frame, 20);
    const by = sampleSpline(table.ball.y, frame, 20);
    const bz = sampleAt(table.ball.z, frame, 0);
    const ballOk = !Number.isNaN(bx) && !Number.isNaN(by);
    const ballH = Number.isNaN(bz) ? 0 : Math.max(0, bz);

    let ballSpd = 0;
    if (ballOk && !Number.isNaN(s.lastBallX)) {
      ballSpd = Math.hypot(bx - s.lastBallX, by - s.lastBallZ);
    }
    if (ballOk) { s.lastBallX = bx; s.lastBallZ = by; }

    // ── Player centroid + spread ──
    let sx = 0, sy = 0, n = 0;
    let mnX = Infinity, mxX = -Infinity;
    for (const trk of table.tracks) {
      if (trk.teamId !== 0 && trk.teamId !== 1) continue;
      const x = sampleSpline(trk.x, frame);
      const y = sampleSpline(trk.y, frame);
      if (Number.isNaN(x) || Number.isNaN(y)) continue;
      sx += x; sy += y; n++;
      if (x < mnX) mnX = x;
      if (x > mxX) mxX = x;
    }
    const centX = n > 0 ? sx / n : PITCH_LENGTH / 2;
    const centY = n > 0 ? sy / n : PITCH_WIDTH / 2;
    const spread = n > 1 ? mxX - mnX : 40;

    // ── Find nearest player to ball (for POV anchor) ──
    let nearDist = Infinity;
    let nearPx = centX, nearPy = centY;
    let nearFwdX = 0, nearFwdZ = -1;
    if (ballOk) {
      for (const trk of table.tracks) {
        if (trk.teamId !== 0 && trk.teamId !== 1) continue;
        const tx = sampleSpline(trk.x, frame);
        const ty = sampleSpline(trk.y, frame);
        if (Number.isNaN(tx) || Number.isNaN(ty)) continue;
        const d = Math.hypot(tx - bx, ty - by);
        if (d < nearDist) {
          nearDist = d;
          nearPx = tx;
          nearPy = ty;
          // Forward direction from velocity
          const nx = sampleSpline(trk.x, frame + 1);
          const ny = sampleSpline(trk.y, frame + 1);
          if (!Number.isNaN(nx) && !Number.isNaN(ny)) {
            const len = Math.hypot(nx - tx, ny - ty);
            if (len > 0.01) {
              nearFwdX = (nx - tx) / len;
              nearFwdZ = (ny - ty) / len;
            }
          }
        }
      }
    }

    // ── Mode switching ──
    const elapsed = frame - s.lastSwitch;
    const forcePovId = povPlayerId; // Explicit external POV request
    
    if (forcePovId) {
      s.mode = "pov";
    } else if (s.mode === "broadcast" && nearDist < 3.0 && elapsed > MIN_HOLD) {
      s.mode = "pov";
      s.lastSwitch = frame;
    } else if (s.mode === "pov" && (nearDist > 8.0 || ballH > 5.0) && elapsed > MIN_HOLD) {
      s.mode = "broadcast";
      s.lastSwitch = frame;
    }

    // ── Compute desired camera position + lookAt ──
    let desired: THREE.Vector3;
    let lookAt: THREE.Vector3;

    if (s.mode === "pov") {
      let anchorX = nearPx, anchorY = nearPy, fwdX = nearFwdX, fwdZ = nearFwdZ;
      
      // If forced, find that specific track
      if (forcePovId) {
        const trk = table.tracks.find(t => String(t.trackId) === String(forcePovId));
        if (trk) {
           const tx = sampleSpline(trk.x, frame);
           const ty = sampleSpline(trk.y, frame);
           if (!Number.isNaN(tx)) {
             anchorX = tx; anchorY = ty;
             const h = sampleSplineAngle(trk.heading, frame);
             if (!Number.isNaN(h)) {
                // Heading to vector
                const rad = -(h * Math.PI) / 180;
                fwdX = Math.cos(rad);
                fwdZ = -Math.sin(rad); // Negated since -Z is forward in Three
             }
           }
        }
      }

      const [pwx, pwz] = pitchToWorld(anchorX, anchorY);
      const [bwx, bwz] = pitchToWorld(bx, by);
      
      // Position at eye level, subtly behind the model's head to avoid clipping
      desired = new THREE.Vector3(
        pwx - fwdX * POV_BEHIND,
        HEAD_H,
        pwz - fwdZ * POV_BEHIND,
      );
      lookAt = ballOk ? new THREE.Vector3(bwx, ballH, bwz) : new THREE.Vector3(pwx + fwdX * 10, HEAD_H, pwz + fwdZ * 10);
    } else {
      // Broadcast — ball-aware blend
      const airW = Math.min(1, ballH / 1.5) * 0.7;
      const spdW = Math.min(1, ballSpd / 0.5) * 0.55;
      const bw = Math.max(airW, spdW);
      s.ballFollowW = s.ballFollowW * 0.92 + bw * 0.08;

      const tx = ballOk ? centX * (1 - s.ballFollowW) + bx * s.ballFollowW : centX;
      const ty = ballOk ? centY * (1 - s.ballFollowW) + by * s.ballFollowW : centY;
      const [twx, twz] = pitchToWorld(tx, ty);

      s.targetX = s.targetX * 0.94 + twx * 0.06;
      s.targetZ = s.targetZ * 0.94 + twz * 0.06;

      const sn = Math.min(1, Math.max(0, (spread - 20) / 40));
      const h = 32 + sn * 16;
      const d = 42 + sn * 22;

      desired = new THREE.Vector3(s.targetX + 6, h, s.targetZ + d);

      // Lead the camera 5m ahead of ball velocity vector
      let leadX = 0, leadZ = 0;
      if (ballOk && !Number.isNaN(s.lastBallX)) {
        const vx = bx - s.lastBallX;
        const vz = by - s.lastBallZ;
        const vmag = Math.hypot(vx, vz);
        if (vmag > 0.01) {
          const [lwx, lwz] = pitchToWorld(bx + (vx / vmag) * BALL_LEAD, by + (vz / vmag) * BALL_LEAD);
          const [bwxC, bwzC] = pitchToWorld(bx, by);
          leadX = (lwx - bwxC) * Math.min(1, vmag * 2); // fade in with speed
          leadZ = (lwz - bwzC) * Math.min(1, vmag * 2);
        }
      }

      const lookY = 1.2 + (ballOk ? Math.min(ballH * 0.5, 3) : 0);
      lookAt = new THREE.Vector3(s.targetX + leadX, lookY, s.targetZ + leadZ);
    }

    // ── Smooth + apply ──
    const posA = s.mode === "pov" ? 0.2 : 0.05;
    const lookA = s.mode === "pov" ? 0.35 : 0.06; // More lag on look target
    s.pos.lerp(desired, posA);
    s.look.lerp(lookAt, lookA);

    camera.position.copy(s.pos);
    camera.lookAt(s.look);

    // FOV shift — wider on POV, cinematic tight 'long lens' on broadcast
    const sn = Math.min(1, Math.max(0, (spread - 15) / 45));
    const bCastFov = 22 + sn * 15; // Tighter focal length
    const tgtFov = s.mode === "pov" ? 82 : bCastFov;
    s.fov += (tgtFov - s.fov) * 0.06;
    
    if (Math.abs(camera.fov - s.fov) > 0.01) {
      camera.fov = s.fov;
      camera.updateProjectionMatrix();
    }
  });

  return null;
};
