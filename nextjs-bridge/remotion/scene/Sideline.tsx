import { PITCH_LENGTH, PITCH_WIDTH } from "./constants";

const H = 1.2;
const DEPTH = 0.3;
const HOME = "#c6004a";
const AWAY = "#003c71";

export const Sideline: React.FC = () => (
  <group>
    {/* Home side — -Z */}
    <mesh position={[0, H / 2, -(PITCH_WIDTH / 2 + DEPTH / 2)]} receiveShadow>
      <boxGeometry args={[PITCH_LENGTH, H, DEPTH]} />
      <meshStandardMaterial color={HOME} emissive={HOME} emissiveIntensity={0.6} roughness={0.5} metalness={0.1} />
    </mesh>
    {/* Away side — +Z */}
    <mesh position={[0, H / 2, PITCH_WIDTH / 2 + DEPTH / 2]} receiveShadow>
      <boxGeometry args={[PITCH_LENGTH, H, DEPTH]} />
      <meshStandardMaterial color={AWAY} emissive={AWAY} emissiveIntensity={0.6} roughness={0.5} metalness={0.1} />
    </mesh>
  </group>
);
