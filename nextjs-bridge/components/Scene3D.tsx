"use client";

import React, { Suspense, Component } from "react";
import { Canvas } from "@react-three/fiber";
import { Stage } from "../remotion/scene/Stage";
import type { ReplayTable } from "../remotion/data/types";

type Props = {
  table: ReplayTable;
  frame: number;
};

class SceneErrorBoundary extends Component<
  { children: React.ReactNode },
  { error: string | null }
> {
  state = { error: null };
  static getDerivedStateFromError(e: Error) {
    return { error: e.message };
  }
  render() {
    if (this.state.error) {
      return (
        <div style={{ color: "#f87171", padding: 16, fontFamily: "monospace", fontSize: 12 }}>
          Scene error: {this.state.error}
        </div>
      );
    }
    return this.props.children;
  }
}

/**
 * A standalone Three.js viewer for the FC26 Sandbox.
 * Wraps the Stage in a standard R3F Canvas for the Next.js landing page.
 */
export const Scene3D: React.FC<Props> = ({ table, frame }) => {
  return (
    <div style={{ width: "100%", height: "400px", borderRadius: "12px", overflow: "hidden", background: "#05070a" }}>
      <SceneErrorBoundary>
        <Canvas
          shadows
          camera={{ position: [0, 50, 80], fov: 50 }}
          gl={{ antialias: true, toneMapping: 4 }}
        >
          <Suspense fallback={null}>
            <Stage table={table} frame={frame} videoFps={table.fps} />
          </Suspense>
        </Canvas>
      </SceneErrorBoundary>
    </div>
  );
};
