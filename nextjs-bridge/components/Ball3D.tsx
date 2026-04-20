'use client';

import { useFrame } from '@react-three/fiber';
import { useRef, useState, useMemo } from 'react';
import * as THREE from 'three';
import { Trail, Float } from '@react-three/drei';

interface TrajectoryEntry {
  x: number;
  y: number;
  z: number;
  frameIndex?: number;
}

interface DecisionPoint {
  frameIndex: number;
  label: string;
}

interface Ball3DProps {
  trajectoryData: TrajectoryEntry[];
  fps?: number;
  decisionPoints?: DecisionPoint[];
  onDecisionPointReached?: (label: string) => void;
  pitchWidth?: number;
  pitchHeight?: number;
}

/**
 * Ball3D Component — Professional Edition
 * 
 * Provides interactive replay capabilities, including decision point pauses,
 * motion trails, and realistic physics-corrected coordinate mapping.
 */
export const Ball3D = ({ 
  trajectoryData, 
  fps = 30, 
  decisionPoints = [], 
  onDecisionPointReached,
  pitchWidth = 105,
  pitchHeight = 68
}: Ball3DProps) => {
  const ballRef = useRef<THREE.Mesh>(null);
  const [isPaused, setIsPaused] = useState(false);
  const reachedPoints = useRef<Set<number>>(new Set());

  // Coordinate Normalization: map (0,0) at corner to (0,0) at centre circle
  const center_x = pitchWidth / 2;
  const center_y = pitchHeight / 2;

  useFrame((state) => {
    if (!trajectoryData || trajectoryData.length === 0 || isPaused) return;

    const time = state.clock.getElapsedTime();
    const frameIndex = Math.floor((time * fps) % trajectoryData.length);
    const data = trajectoryData[frameIndex];
    
    // Check for Decision Points
    const dp = decisionPoints.find(p => p.frameIndex === frameIndex);
    if (dp && !reachedPoints.current.has(dp.frameIndex)) {
      reachedPoints.current.add(dp.frameIndex);
      setIsPaused(true);
      if (onDecisionPointReached) onDecisionPointReached(dp.label);
      return;
    }

    if (ballRef.current && data) {
      const { x, y, z } = data;
      
      // Coordinate Mapping:
      // Three.js (0,0,0) is usually center of the pitch.
      // Python (0,0) is top-left corner.
      // x -> (data.x - center_x)
      // z -> data.z (hallucinated height)
      // y -> -(data.y - center_y) (Three.js Z is depth)
      
      ballRef.current.position.set(
        x - center_x,
        z,
        -(y - center_y)
      );
    }
  });

  return (
    <group>
      {/* Motion Trail for visual height/arc perception */}
      <Trail
        width={0.5}
        length={10}
        color={new THREE.Color("#00d4aa")}
        attenuation={(t) => t * t}
      >
        <mesh ref={ballRef} castShadow receiveShadow>
          <sphereGeometry args={[0.11, 32, 32]} />
          <meshStandardMaterial 
            color="#ffffff" 
            roughness={0.2} 
            metalness={0.1}
          />
        </mesh>
      </Trail>

      {/* Interactive decision indicator (optional/extendable) */}
      {isPaused && (
        <Float speed={2} rotationIntensity={0.5} floatIntensity={0.5}>
          <mesh position={[ballRef.current?.position.x || 0, (ballRef.current?.position.y || 0) + 1, ballRef.current?.position.z || 0]}>
            <coneGeometry args={[0.2, 0.4, 3]} />
            <meshBasicMaterial color="#00d4aa" />
          </mesh>
        </Float>
      )}
    </group>
  );
};
