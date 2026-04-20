import { useRef, useMemo, useEffect } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import { useGLTF, useAnimations, Html } from "@react-three/drei";
import { SkeletonUtils } from "three-stdlib";
import { staticFile } from "remotion";
import * as THREE from "three";
import type { TrackTrajectory, BallTrajectory, MocapLabel } from "../data/types";
import { PLAYER_HEIGHT, pitchToWorld } from "./constants";

const AVATAR_URL = staticFile("/avatars/player.glb");
try { useGLTF.preload(AVATAR_URL); } catch { /* noop */ }

// ── Constants ──
const LEAN_MAX_RAD = (15 * Math.PI) / 180;
const LEAN_ONSET = 1.0;
const LEAN_FULL = 6.0;
const KICK_DIST = 1.8;
const KICK_BALL_SPEED = 6;
const KICK_COOLDOWN = 15;
const KICK_DURATION = 6;
const ROT_LPF_FRAMES = 5;

// Shared frustum — one per data-frame
const _frustum = new THREE.Frustum();
const _mat4 = new THREE.Matrix4();
let _frustumFrame = -1;

// Map backend mocap labels → GLB clip names
const CLIP_MAP: Record<MocapLabel, string> = {
  idle: "Idle",
  run: "Running",
  sprint: "Running",
  dribble: "Running",
  kick: "Idle",
};

// timeScale per mocap label
function getTimeScale(label: MocapLabel, speedMs: number): number {
  switch (label) {
    case "sprint": return Math.min(2.5, Math.max(1.2, speedMs / 3.0));
    case "dribble": return Math.min(1.5, Math.max(0.4, speedMs / 4.0));
    case "run": return Math.min(2.0, Math.max(0.5, speedMs / 3.5));
    default: return 1.0;
  }
}

// Rigid body lock threshold — no procedural bones below this speed
const RIGID_BODY_SPEED_KMH = 1.0;

type Props = {
  track: TrackTrajectory;
  ball: BallTrajectory;
  teamColor: string;
  frame: number;
  /** Force-teleport to position (skip LPF/smoothing). */
  teleport?: boolean;
};

/**
 * One PlayerRig per trackId. Position read DIRECTLY from track.x[frame].
 * Animation state from track.mocapState[frame]. No interpolation on position.
 * trackId = player_id from video. If video says ID 7 sprints, this rig sprints.
 */
export const PlayerRig: React.FC<Props> = ({ track, ball, teamColor, frame, teleport }) => {
  const group = useRef<THREE.Group>(null!);
  const hudRef = useRef<THREE.Group>(null!);
  const arrowRef = useRef<THREE.Group>(null!);
  const { scene, animations } = useGLTF(AVATAR_URL);
  const camera = useThree((s) => s.camera);

  const clone = useMemo(() => SkeletonUtils.clone(scene), [scene]);
  const { actions } = useAnimations(animations, clone);

  const bones = useMemo(() => ({
    rightLeg: clone.getObjectByName("mixamorigRightUpLeg") as THREE.Bone | null,
    head: clone.getObjectByName("mixamorigHead") as THREE.Bone | null,
    spine: clone.getObjectByName("mixamorigSpine1") as THREE.Bone | null,
    spine2: clone.getObjectByName("mixamorigSpine2") as THREE.Bone | null,
  }), [clone]);

  // Persistent state
  const prevPos = useRef(new THREE.Vector3());
  const prevBallWorld = useRef(new THREE.Vector3());
  const prevSpeed = useRef(0);
  const activeClip = useRef<string>("Idle");
  const kickTimer = useRef(0);
  const lastKickFrame = useRef(-100);
  const spineBaseX = useRef(0);
  const rotBuffer = useRef<number[]>([]);

  // ── Init ──
  useEffect(() => {
    clone.traverse((obj) => {
      if ((obj as THREE.Mesh).isMesh) {
        const mat = (obj as THREE.Mesh).material as THREE.MeshStandardMaterial;
        if (mat?.color) mat.color.set(teamColor);
        obj.castShadow = true;
      }
    });
    if (bones.spine) spineBaseX.current = bones.spine.rotation.x;
    const idle = actions.Idle;
    if (idle) { idle.reset().play(); idle.weight = 1; }
  }, [clone, actions, teamColor, bones.spine]);

  // ── Clamp frame to array bounds ──
  const sf = Math.max(0, Math.min(frame, track.x.length - 1));

  // ── IDENTITY SYNC: mocapState[frame] drives animation ──
  const mocapLabel: MocapLabel = track.mocapState[sf] ?? "idle";

  useEffect(() => {
    const targetClip = CLIP_MAP[mocapLabel] ?? "Idle";
    if (targetClip === activeClip.current) return;

    const fadeOut = actions[activeClip.current];
    const fadeIn = actions[targetClip];
    if (fadeOut && fadeIn && fadeOut !== fadeIn) {
      fadeOut.fadeOut(0.25);
      fadeIn.reset().fadeIn(0.25).play();
    } else if (fadeIn) {
      fadeIn.reset().play();
    }
    activeClip.current = targetClip;
  }, [mocapLabel, actions]);

  // ── Per-render-frame ──
  useFrame((_, dt) => {
    if (!group.current || dt <= 0) return;

    // ── POSITION: Direct array index. track.x[frame] = video frame_id. ──
    const px = track.x[sf];
    const py = track.y[sf];
    if (px === undefined || py === undefined || Number.isNaN(px) || Number.isNaN(py)) {
      group.current.visible = false;
      return;
    }
    const [wx, wz] = pitchToWorld(px, py);
    group.current.position.set(wx, 0, wz);
    const pos = group.current.position;

    // ── Frustum cull ──
    if (_frustumFrame !== frame) {
      _mat4.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
      _frustum.setFromProjectionMatrix(_mat4);
      _frustumFrame = frame;
    }
    if (!_frustum.containsPoint(pos)) {
      group.current.visible = false;
      prevPos.current.copy(pos);
      return;
    }
    group.current.visible = true;

    // ── Velocity from backend vectors (direct index) ──
    const vx = track.vectorX[sf];
    const vy = track.vectorY[sf];
    const hasVec = vx !== undefined && vy !== undefined && !Number.isNaN(vx) && !Number.isNaN(vy);
    const vecSpeed = hasVec ? Math.hypot(vx, vy) : 0;
    const posDelta = dt > 0 ? pos.distanceTo(prevPos.current) / dt : 0;
    const speedMs = hasVec ? vecSpeed : posDelta;
    const smoothSpeed = prevSpeed.current * 0.7 + speedMs * 0.3;
    prevSpeed.current = smoothSpeed;

    // ── Ball (direct index) ──
    const ballPx = ball.x[sf];
    const ballPy = ball.y[sf];
    const ballPz = ball.z[sf];
    const ballOk = ballPx !== undefined && ballPy !== undefined && !Number.isNaN(ballPx) && !Number.isNaN(ballPy);
    const ballWorld = new THREE.Vector3();
    if (ballOk) {
      const [bwx, bwz] = pitchToWorld(ballPx, ballPy);
      ballWorld.set(bwx, (ballPz !== undefined && !Number.isNaN(ballPz)) ? Math.max(0, ballPz) : 0, bwz);
    }

    // ── Kick detection ──
    if (ballOk) {
      const ballSpd = dt > 0 ? ballWorld.distanceTo(prevBallWorld.current) / dt : 0;
      const distBall = pos.distanceTo(ballWorld);
      if (ballSpd > KICK_BALL_SPEED && distBall < KICK_DIST && sf - lastKickFrame.current > KICK_COOLDOWN) {
        kickTimer.current = 1.0;
        lastKickFrame.current = sf;
      }
    }

    // ══════════════════════════════════════════════
    //  ORIENTATION — 5-frame Low-Pass Filter
    //  (bypassed on teleport — snap immediately)
    // ══════════════════════════════════════════════

    let targetYaw = group.current.rotation.y;

    // P1: Backend velocity vector
    if (hasVec && vecSpeed > 0.3) {
      targetYaw = Math.atan2(vx, -vy);
    }
    // P2: Backend heading (direct index)
    else {
      const heading = track.heading[sf];
      if (heading !== undefined && !Number.isNaN(heading)) {
        targetYaw = -(heading * Math.PI) / 180;
      }
      // P3: Position delta
      else if (posDelta > 0.2) {
        const dx = pos.x - prevPos.current.x;
        const dz = pos.z - prevPos.current.z;
        targetYaw = Math.atan2(dx, dz);
        if (ballOk) {
          const ballYaw = Math.atan2(ballWorld.x - pos.x, ballWorld.z - pos.z);
          const d = pos.distanceTo(ballWorld);
          const w = Math.max(0, 1 - d / 12) * 0.6;
          targetYaw = lerpAngle(targetYaw, ballYaw, w);
        }
      }
      // P4: Idle → face ball
      else if (ballOk && pos.distanceTo(ballWorld) > 0.5) {
        targetYaw = Math.atan2(ballWorld.x - pos.x, ballWorld.z - pos.z);
      }
    }

    if (teleport) {
      // THE ANCHOR: snap rotation immediately, flush LPF buffer
      group.current.rotation.y = targetYaw;
      rotBuffer.current = [targetYaw];
    } else {
      // Ring buffer circular mean
      const buf = rotBuffer.current;
      buf.push(targetYaw);
      if (buf.length > ROT_LPF_FRAMES) buf.shift();
      let sinSum = 0, cosSum = 0;
      for (const a of buf) { sinSum += Math.sin(a); cosSum += Math.cos(a); }
      group.current.rotation.y = Math.atan2(sinSum / buf.length, cosSum / buf.length);
    }

    // ── Animation timeScale ──
    const runAction = actions.Running;
    if (runAction && (mocapLabel === "run" || mocapLabel === "sprint" || mocapLabel === "dribble")) {
      runAction.timeScale = getTimeScale(mocapLabel, smoothSpeed);
    }

    // ══════════════════════════════════════════════
    //  PROCEDURAL BONE OVERRIDES
    //  RIGID BODY LOCK: disabled unless speed_kmh > 1.0
    // ══════════════════════════════════════════════

    const speedKmh = track.speedKmh[sf];
    const isMoving = speedKmh !== undefined && !Number.isNaN(speedKmh) && speedKmh > RIGID_BODY_SPEED_KMH;

    if (isMoving) {
      // Spine lean
      if (bones.spine) {
        const speedFactor = Math.max(0, (smoothSpeed - LEAN_ONSET) / (LEAN_FULL - LEAN_ONSET));
        const leanRad = Math.min(LEAN_MAX_RAD, speedFactor * LEAN_MAX_RAD);
        const currentLean = bones.spine.rotation.x - spineBaseX.current;
        const smoothedLean = currentLean + (-leanRad - currentLean) * 0.12;
        bones.spine.rotation.x = spineBaseX.current + smoothedLean;
        if (bones.spine2) bones.spine2.rotation.x += smoothedLean * 0.4;
      }

      // Procedural kick
      if (kickTimer.current > 0) {
        kickTimer.current -= dt * KICK_DURATION;
        if (bones.rightLeg) {
          bones.rightLeg.rotation.x -= Math.sin(kickTimer.current * Math.PI) * 1.2;
        }
      }

      // Head tracking (anti-Exorcist: normalize to [-π,π] before clamping)
      if (bones.head && ballOk) {
        const pitchTarget = Math.max(-0.5, Math.min(0.5, (ballWorld.y - 1.5) * 0.15));
        bones.head.rotation.x += (pitchTarget - bones.head.rotation.x) * 0.08;
        const worldBallAngle = Math.atan2(ballWorld.x - pos.x, ballWorld.z - pos.z);
        let headYaw = worldBallAngle - group.current.rotation.y;
        while (headYaw > Math.PI) headYaw -= Math.PI * 2;
        while (headYaw < -Math.PI) headYaw += Math.PI * 2;
        headYaw = Math.max(-1.4, Math.min(1.4, headYaw));
        bones.head.rotation.y += (headYaw - bones.head.rotation.y) * 0.06;
      }
    } else {
      // RIGID BODY: reset spine lean to rest pose
      if (bones.spine) {
        const currentLean = bones.spine.rotation.x - spineBaseX.current;
        bones.spine.rotation.x = spineBaseX.current + currentLean * 0.9; // ease back to rest
      }
      kickTimer.current = 0;
    }

    // HUD + arrow
    if (hudRef.current) hudRef.current.position.set(pos.x, PLAYER_HEIGHT + 0.6, pos.z);
    if (arrowRef.current) {
      if (hasVec && vecSpeed > 0.5) {
        arrowRef.current.position.set(pos.x, 0.15, pos.z);
        arrowRef.current.rotation.y = -Math.atan2(vy, vx);
        arrowRef.current.visible = true;
      } else {
        arrowRef.current.visible = false;
      }
    }

    prevPos.current.copy(pos);
    if (ballOk) prevBallWorld.current.copy(ballWorld);
  });

  // ── HUD data (direct index) ──
  const speedVal = track.speedKmh[sf];
  const hudVx = track.vectorX[sf];
  const hudVy = track.vectorY[sf];
  const hasSpeed = speedVal !== undefined && !Number.isNaN(speedVal);
  const hasVector = hudVx !== undefined && hudVy !== undefined && !Number.isNaN(hudVx) && !Number.isNaN(hudVy);

  const speedColor = !hasSpeed ? "#fff"
    : speedVal < 8 ? "#ffffff"
      : speedVal < 20 ? "#00d4aa"
        : speedVal < 28 ? "#ffaa00"
          : "#ff3333";

  return (
    <group>
      <primitive ref={group} object={clone} />

      {hasSpeed && (
        <group ref={hudRef}>
          <Html center distanceFactor={18} style={{ pointerEvents: "none", userSelect: "none" }}>
            <div style={{
              background: "rgba(0,0,0,0.75)",
              border: `1px solid ${speedColor}`,
              borderRadius: 6,
              padding: "2px 8px",
              whiteSpace: "nowrap",
              fontFamily: "monospace",
              fontSize: 11,
              color: speedColor,
              textAlign: "center",
              lineHeight: 1.3,
              backdropFilter: "blur(4px)",
            }}>
              <div style={{ fontWeight: 700 }}>
                #{track.trackId} {speedVal.toFixed(1)} <span style={{ fontSize: 9, opacity: 0.7 }}>km/h</span>
              </div>
              {hasVector && (
                <div style={{ fontSize: 9, opacity: 0.6, marginTop: 1 }}>
                  [{hudVx.toFixed(1)}, {hudVy.toFixed(1)}]
                </div>
              )}
            </div>
          </Html>
        </group>
      )}

      {hasVector && (
        <group ref={arrowRef}>
          <mesh rotation={[0, 0, -Math.PI / 2]}>
            <coneGeometry args={[0.25, Math.min(Math.hypot(hudVx, hudVy) * 0.3, 3), 6]} />
            <meshBasicMaterial color={speedColor} transparent opacity={0.6} />
          </mesh>
        </group>
      )}
    </group>
  );
};

function lerpAngle(a: number, b: number, t: number): number {
  let diff = b - a;
  while (diff < -Math.PI) diff += Math.PI * 2;
  while (diff > Math.PI) diff -= Math.PI * 2;
  return a + diff * t;
}
