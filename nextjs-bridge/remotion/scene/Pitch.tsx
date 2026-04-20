import * as THREE from "three";
import { useMemo, useRef } from "react";
import { useLoader, useFrame, extend } from "@react-three/fiber";
import { shaderMaterial } from "@react-three/drei";
import { staticFile } from "remotion";
import {
  PITCH_LENGTH,
  PITCH_WIDTH,
  CHALK_COLOR,
} from "./constants";

/* ================================================================
 * GRASS INTERACTION SHADER
 *
 * Standard grass PBR + up to 24 player foot positions that create
 * subtle dark divots (color darken + roughness increase).
 * ================================================================ */

const MAX_FEET = 24;

const GrassInteractionMaterial = shaderMaterial(
  {
    colorMap: null as THREE.Texture | null,
    normalMap: null as THREE.Texture | null,
    roughMap: null as THREE.Texture | null,
    footPositions: new Float32Array(MAX_FEET * 2), // [x,z, x,z, ...]
    footCount: 0,
  },
  // Vertex
  /* glsl */ `
    varying vec2 vUv;
    varying vec3 vWorldPos;
    varying vec3 vNormal;
    void main() {
      vUv = uv;
      vec4 wp = modelMatrix * vec4(position, 1.0);
      vWorldPos = wp.xyz;
      vNormal = normalize(normalMatrix * normal);
      gl_Position = projectionMatrix * viewMatrix * wp;
    }
  `,
  // Fragment
  /* glsl */ `
    uniform sampler2D colorMap;
    uniform sampler2D normalMap;
    uniform sampler2D roughMap;
    uniform float footPositions[${MAX_FEET * 2}];
    uniform int footCount;

    varying vec2 vUv;
    varying vec3 vWorldPos;
    varying vec3 vNormal;

    void main() {
      vec4 baseColor = texture2D(colorMap, vUv);
      float baseRough = texture2D(roughMap, vUv).r * 0.8;

      // Accumulate foot influence
      float footInfluence = 0.0;
      for (int i = 0; i < ${MAX_FEET}; i++) {
        if (i >= footCount) break;
        float fx = footPositions[i * 2];
        float fz = footPositions[i * 2 + 1];
        float d = distance(vWorldPos.xz, vec2(fx, fz));
        // Soft radial falloff: full effect at 0m, fades to 0 at 0.8m
        footInfluence += smoothstep(0.8, 0.0, d) * 0.35;
      }
      footInfluence = clamp(footInfluence, 0.0, 0.5);

      // Darken + increase roughness at foot positions
      vec3 color = baseColor.rgb * (1.0 - footInfluence * 0.4);
      float roughness = baseRough + footInfluence * 0.3;

      // Simple lighting (let Three.js handle the rest via tonemap)
      vec3 lightDir = normalize(vec3(0.3, 1.0, 0.2));
      float diff = max(dot(vNormal, lightDir), 0.0) * 0.6 + 0.4;
      gl_FragColor = vec4(color * diff, 1.0);
    }
  `
);

extend({ GrassInteractionMaterial });

/**
 * PBR grass pitch (105x68 m) with tiled textures, chalk lines,
 * and per-player foot divot shader.
 */
type PitchProps = {
  onPitchClick?: (x: number, z: number) => void;
  /** World-space [x,z] pairs for each player's feet (up to 24). */
  footPositions?: Float32Array;
  footCount?: number;
};

export const Pitch: React.FC<PitchProps> = ({
  onPitchClick,
  footPositions: footPos,
  footCount = 0,
}) => {
  const matRef = useRef<any>(null!);

  const [colorMap, normalMap, roughMap] = useLoader(THREE.TextureLoader, [
    staticFile("/textures/grass/color.jpg"),
    staticFile("/textures/grass/normal.jpg"),
    staticFile("/textures/grass/rough.jpg"),
  ]);

  useMemo(() => {
    for (const t of [colorMap, normalMap, roughMap]) {
      t.wrapS = t.wrapT = THREE.RepeatWrapping;
      t.repeat.set(24, 16);
      t.anisotropy = 8;
    }
    colorMap.colorSpace = THREE.SRGBColorSpace;
  }, [colorMap, normalMap, roughMap]);

  // Push foot positions into shader uniform — single upload, no extra draw calls.
  // The uniform float[] is already a flat array in GLSL; we just memcpy the
  // shared Float32Array directly into the uniform backing store each frame.
  const lastFootCount = useRef(0);
  useFrame(() => {
    if (!matRef.current) return;
    const mat = matRef.current;
    // Only upload when data actually changed (count or positions)
    if (footCount !== lastFootCount.current) {
      mat.footCount = footCount;
      lastFootCount.current = footCount;
    }
    if (footPos && footPos.length > 0) {
      // Direct write into the uniform's backing Float32Array — zero allocations
      const dst = mat.footPositions as Float32Array;
      const len = Math.min(footPos.length, MAX_FEET * 2);
      for (let i = 0; i < len; i++) dst[i] = footPos[i];
    }
  });

  return (
    <group>
      {/* Grass surface with interaction shader */}
      <mesh
        rotation={[-Math.PI / 2, 0, 0]}
        receiveShadow
        onPointerDown={(e) => {
          if (onPitchClick) onPitchClick(e.point.x, e.point.z);
        }}
      >
        <planeGeometry args={[PITCH_LENGTH + 10, PITCH_WIDTH + 10]} />
        {/* @ts-ignore — extended material */}
        <grassInteractionMaterial
          ref={matRef}
          colorMap={colorMap}
          normalMap={normalMap}
          roughMap={roughMap}
          footCount={footCount}
        />
      </mesh>

      {/* Chalk lines as mesh overlay */}
      <PitchLines />
    </group>
  );
};

/* ── Chalk line overlay meshes at y=0.01 ── */

const LW = 0.12; // line width metres
const LINE_MAT_PROPS = {
  color: CHALK_COLOR,
  emissive: CHALK_COLOR,
  emissiveIntensity: 0.3,
  roughness: 0.9,
  metalness: 0,
} as const;

const L = PITCH_LENGTH;
const W = PITCH_WIDTH;
const Y = 0.01;

const PitchLines: React.FC = () => (
  <group>
    {/* Outer boundary */}
    <LineBox x={0} z={0} w={L} d={LW} />           {/* bottom */}
    <LineBox x={0} z={W} w={L} d={LW} />           {/* top */}
    <LineBox x={0} z={W / 2} w={LW} d={W} />       {/* left */}
    <LineBox x={L} z={W / 2} w={LW} d={W} />       {/* right */}

    {/* Halfway line */}
    <LineBox x={L / 2} z={W / 2} w={LW} d={W} />

    {/* Centre circle (approximated with 48 segments) */}
    <CentreCircle />

    {/* Centre spot */}
    <mesh position={[L / 2 - L / 2, Y, W / 2 - W / 2]}>
      <cylinderGeometry args={[0.25, 0.25, 0.005, 16]} />
      <meshStandardMaterial {...LINE_MAT_PROPS} />
    </mesh>

    {/* Penalty areas — 16.5 × 40.3 */}
    <PenaltyBox side="left" />
    <PenaltyBox side="right" />

    {/* Six-yard boxes — 5.5 × 18.3 */}
    <SixYardBox side="left" />
    <SixYardBox side="right" />

    {/* Penalty spots */}
    <mesh position={[11 - L / 2, Y, 0]}>
      <cylinderGeometry args={[0.25, 0.25, 0.005, 16]} />
      <meshStandardMaterial {...LINE_MAT_PROPS} />
    </mesh>
    <mesh position={[L - 11 - L / 2, Y, 0]}>
      <cylinderGeometry args={[0.25, 0.25, 0.005, 16]} />
      <meshStandardMaterial {...LINE_MAT_PROPS} />
    </mesh>
  </group>
);

/** A thin box line segment centered at pitch coords (x, z), world-centered */
const LineBox: React.FC<{ x: number; z: number; w: number; d: number }> = ({ x, z, w, d }) => (
  <mesh position={[x - L / 2, Y, z - W / 2]}>
    <boxGeometry args={[w, 0.005, d]} />
    <meshStandardMaterial {...LINE_MAT_PROPS} />
  </mesh>
);

/** Rectangular outline from 4 LineBoxes */
const RectOutline: React.FC<{ px: number; pz: number; rw: number; rh: number }> = ({ px, pz, rw, rh }) => (
  <group>
    <LineBox x={px} z={pz + rh / 2} w={LW} d={rh} />           {/* left edge */}
    <LineBox x={px + rw} z={pz + rh / 2} w={LW} d={rh} />      {/* right edge */}
    <LineBox x={px + rw / 2} z={pz} w={rw} d={LW} />           {/* bottom */}
    <LineBox x={px + rw / 2} z={pz + rh} w={rw} d={LW} />      {/* top */}
  </group>
);

const PenaltyBox: React.FC<{ side: "left" | "right" }> = ({ side }) => {
  const penY = (W - 40.3) / 2;
  const px = side === "left" ? 0 : L - 16.5;
  return <RectOutline px={px} pz={penY} rw={16.5} rh={40.3} />;
};

const SixYardBox: React.FC<{ side: "left" | "right" }> = ({ side }) => {
  const sixY = (W - 18.3) / 2;
  const px = side === "left" ? 0 : L - 5.5;
  return <RectOutline px={px} pz={sixY} rw={5.5} rh={18.3} />;
};

/** Centre circle from thin box segments */
const CentreCircle: React.FC = () => {
  const segs = 48;
  const r = 9.15;
  const pieces: React.ReactNode[] = [];
  for (let i = 0; i < segs; i++) {
    const a0 = (i / segs) * Math.PI * 2;
    const a1 = ((i + 1) / segs) * Math.PI * 2;
    const cx = (Math.cos(a0) + Math.cos(a1)) / 2 * r;
    const cz = (Math.sin(a0) + Math.sin(a1)) / 2 * r;
    const len = Math.sqrt(
      (Math.cos(a1) * r - Math.cos(a0) * r) ** 2 +
      (Math.sin(a1) * r - Math.sin(a0) * r) ** 2,
    );
    const angle = Math.atan2(
      Math.sin(a1) * r - Math.sin(a0) * r,
      Math.cos(a1) * r - Math.cos(a0) * r,
    );
    pieces.push(
      <mesh key={i} position={[cx, Y, cz]} rotation={[0, -angle, 0]}>
        <boxGeometry args={[len, 0.005, LW]} />
        <meshStandardMaterial {...LINE_MAT_PROPS} />
      </mesh>,
    );
  }
  return <group>{pieces}</group>;
};
