"use client";

import React, { Suspense } from "react";
import { Canvas } from "@react-three/fiber";
import { Environment, SoftShadows } from "@react-three/drei";
import { Stage } from "../remotion/scene/Stage";
import type { ReplayTable } from "../remotion/data/types";

type Props = {
  table: ReplayTable;
  frame: number;
};

/**
 * A standalone Three.js viewer for the FC26 Sandbox.
 * Wraps the Stage in a standard R3F Canvas for the Next.js landing page.
 */
export const Scene3D: React.FC<Props> = ({ table, frame }) => {
  return (
    <div style={{ width: "100%", height: "400px", borderRadius: "12px", overflow: "hidden", background: "#05070a" }}>
      <Canvas
        shadows
        camera={{ position: [0, 50, 80], fov: 50 }}
        gl={{ antialias: true, toneMapping: 4 }}
      >
        <Suspense fallback={null}>
          <Stage table={table} frame={frame} videoFps={table.fps} />
        </Suspense>
      </Canvas>
    </div>
  );
};
