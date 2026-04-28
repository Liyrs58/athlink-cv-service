import { Environment, SoftShadows, Float, PerspectiveCamera, OrbitControls, Sky, ContactShadows } from "@react-three/drei";
import { useState, useCallback, useRef, useMemo, useEffect } from "react";
import * as THREE from "three";
import { useFrame, useThree } from "@react-three/fiber";
import { EffectComposer, ChromaticAberration, Vignette } from "@react-three/postprocessing";
import type { ReplayTable } from "../data/types";
import { pitchToWorld } from "./constants";
import { Pitch } from "./Pitch";
import { Players } from "./Players";
import { Ball } from "./Ball";
import { Camera } from "./Camera";
import { Sideline } from "./Sideline";
import { PITCH_LENGTH, PITCH_WIDTH } from "./constants";
import { PassingLanes } from "../../components/PassingLanes";

type Props = {
  table: ReplayTable;
  frame: number;
  videoFps: number;
};

const STAND_H = 25;
const STAND_D = 15;
const STAND_COLOR = "#3a3a3a";

export const Stage: React.FC<Props> = ({ table, frame, videoFps }) => {
  const [isDecisionMode, setIsDecisionMode] = useState(false);
  const [simulatedTarget, setSimulatedTarget] = useState<{x: number, z: number} | null>(null);
  const [currentPovId, setCurrentPovId] = useState<string | number | null>(null);
  const [isUserInteracting, setIsUserInteracting] = useState(false);
  const decisionFrame = useRef<number | null>(null);
  const interactionTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastAnchorFrame = useRef(-1);

  // ── FRAME = VIDEO FRAME ID. Clamp to data bounds. No interpolation. ──
  const safeFrame = Math.max(0, Math.min(Math.floor(frame), table.frameCount - 1));

  // ── THE ANCHOR: detect timestamp mismatch → force teleport ──
  // If the JSON timestamp_ms for this frame doesn't match expected playback time,
  // signal children to teleport instead of lerp by bumping the anchor generation.
  const playbackMs = (safeFrame / (videoFps || table.fps)) * 1000;
  const dataMs = table.timestampMs ? table.timestampMs[safeFrame] : playbackMs;
  const drift = Math.abs(playbackMs - dataMs);
  // >100ms drift or frame jump >2 frames = teleport
  const frameJump = Math.abs(safeFrame - lastAnchorFrame.current);
  const mustTeleport = drift > 100 || frameJump > 2;
  lastAnchorFrame.current = safeFrame;

  // Decision mode trigger moved to useEffect — setState during render is illegal
  useEffect(() => {
    if (safeFrame === 200 && !isDecisionMode && !simulatedTarget) {
      setIsDecisionMode(true);
      decisionFrame.current = 200;
    }
  }, [safeFrame, isDecisionMode, simulatedTarget]);

  const handlePitchClick = useCallback((x: number, z: number) => {
    if (isDecisionMode) {
      setSimulatedTarget({ x, z });
      setIsDecisionMode(false);
    }
  }, [isDecisionMode]);

  const handleOrbitStart = useCallback(() => {
    setIsUserInteracting(true);
    if (interactionTimeout.current) clearTimeout(interactionTimeout.current);
  }, []);

  const handleOrbitEnd = useCallback(() => {
    interactionTimeout.current = setTimeout(() => setIsUserInteracting(false), 2000);
  }, []);

  // Decision mode pauses at the decision frame
  const effectiveFrame = isDecisionMode ? (decisionFrame.current ?? safeFrame) : safeFrame;

  // Compute player foot positions for grass shader — direct index, no spline
  const footData = useMemo(() => new Float32Array(48), []);
  const footCount = useMemo(() => {
    let count = 0;
    for (const trk of table.tracks) {
      if (count >= 24) break;
      const px = trk.x[effectiveFrame];
      const py = trk.y[effectiveFrame];
      if (px === undefined || py === undefined || Number.isNaN(px) || Number.isNaN(py)) continue;
      const [wx, wz] = pitchToWorld(px, py);
      footData[count * 2] = wx;
      footData[count * 2 + 1] = wz;
      count++;
    }
    return count;
  }, [table.tracks, effectiveFrame, footData]);

  const camSpeed = useRef(0);
  const lastCamPos = useRef(new THREE.Vector3());

  return (
    <>
      <Environment preset="forest" />
      <Sky distance={450000} sunPosition={[0, 1, 0]} inclination={0} azimuth={0.25} />
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 20, 10]} intensity={1.5} castShadow shadow-mapSize={[1024, 1024]} />

      <SoftShadows size={15} samples={10} focus={0.8} />
      <ContactShadows opacity={0.4} scale={100} blur={2.4} far={10} />

      {isDecisionMode ? (
        <>
          <PerspectiveCamera makeDefault position={[0, 50, 80]} fov={50} />
          <OrbitControls minPolarAngle={0} maxPolarAngle={Math.PI / 2.1} makeDefault onStart={handleOrbitStart} onEnd={handleOrbitEnd} />
        </>
      ) : (
        <>
          {!isUserInteracting && (
            <Camera table={table} frame={effectiveFrame} povPlayerId={currentPovId} />
          )}
          <OrbitControls minPolarAngle={0} maxPolarAngle={Math.PI / 2.1} makeDefault enablePan enableZoom onStart={handleOrbitStart} onEnd={handleOrbitEnd} />
        </>
      )}

      <Pitch onPitchClick={handlePitchClick} footPositions={footData} footCount={footCount} />
      <Sideline />
      <Stands />

      <PassingLanes table={table} frame={effectiveFrame} />
      <Players table={table} frame={effectiveFrame} videoFps={videoFps} teleport={mustTeleport} />

      <Ball table={table} frame={effectiveFrame} simulatedTarget={simulatedTarget} />

      {isDecisionMode && (
        <DecisionOverlay x={0} z={0} />
      )}

      <CameraSpeedFx camSpeed={camSpeed} lastCamPos={lastCamPos} />
    </>
  );
};

const CameraSpeedFx: React.FC<{
  camSpeed: React.MutableRefObject<number>;
  lastCamPos: React.MutableRefObject<THREE.Vector3>;
}> = ({ camSpeed, lastCamPos }) => {
  const { camera } = useThree();
  const offsetRef = useRef(new THREE.Vector2(0, 0));
  const smoothFps = useRef(60);
  const [chromaEnabled, setChromaEnabled] = useState(true);

  useFrame((_, dt) => {
    if (dt <= 0) return;
    const spd = camera.position.distanceTo(lastCamPos.current) / dt;
    camSpeed.current = camSpeed.current * 0.9 + spd * 0.1;
    lastCamPos.current.copy(camera.position);

    const intensity = Math.min(0.003, camSpeed.current * 0.0002);
    offsetRef.current.set(intensity, intensity);

    const instantFps = 1 / dt;
    smoothFps.current = smoothFps.current * 0.95 + instantFps * 0.05;
    const shouldEnable = smoothFps.current >= 30;
    if (shouldEnable !== chromaEnabled) setChromaEnabled(shouldEnable);
  });

  if (chromaEnabled) {
    return (
      <EffectComposer>
        <ChromaticAberration offset={offsetRef.current} radialModulation modulationOffset={0.2} />
        <Vignette darkness={0.35} offset={0.3} />
      </EffectComposer>
    );
  }
  return (
    <EffectComposer>
      <Vignette darkness={0.35} offset={0.3} />
    </EffectComposer>
  );
};

const DecisionOverlay: React.FC<{x: number, z: number}> = ({x, z}) => (
  <Float speed={5} rotationIntensity={0.5} floatIntensity={0.5}>
    <mesh position={[x, 5, z]}>
      <coneGeometry args={[0.5, 1, 4]} />
      <meshBasicMaterial color="#00d4aa" wireframe />
    </mesh>
  </Float>
);

const Stands: React.FC = () => {
  const halfL = PITCH_LENGTH / 2;
  const halfW = PITCH_WIDTH / 2;
  const sidelineOff = halfW + 0.3 + STAND_D / 2 + 2;
  const endOff = halfL + STAND_D / 2 + 2;

  return (
    <group>
      <mesh position={[0, STAND_H / 2, -sidelineOff]} receiveShadow>
        <boxGeometry args={[PITCH_LENGTH + 30, STAND_H, STAND_D]} />
        <meshStandardMaterial color={STAND_COLOR} roughness={0.9} metalness={0} />
      </mesh>
      <mesh position={[0, STAND_H / 2, sidelineOff]} receiveShadow>
        <boxGeometry args={[PITCH_LENGTH + 30, STAND_H, STAND_D]} />
        <meshStandardMaterial color={STAND_COLOR} roughness={0.9} metalness={0} />
      </mesh>
      <mesh position={[-endOff, STAND_H / 2, 0]} receiveShadow>
        <boxGeometry args={[STAND_D, STAND_H, PITCH_WIDTH + 30]} />
        <meshStandardMaterial color={STAND_COLOR} roughness={0.9} metalness={0} />
      </mesh>
      <mesh position={[endOff, STAND_H / 2, 0]} receiveShadow>
        <boxGeometry args={[STAND_D, STAND_H, PITCH_WIDTH + 30]} />
        <meshStandardMaterial color={STAND_COLOR} roughness={0.9} metalness={0} />
      </mesh>
    </group>
  );
};
