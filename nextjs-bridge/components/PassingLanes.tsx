"use client";

import { Line } from "@react-three/drei";
import type { ReplayTable } from "../remotion/data/types";

type Props = {
  table: ReplayTable;
  frame: number;
};

export const PassingLanes: React.FC<Props> = ({ table, frame }) => {
  const bx = table.ball.x[frame];
  const by = table.ball.y[frame];

  // Skip if ball position unknown
  if (isNaN(bx) || isNaN(by)) return null;

  return (
    <group>
      {table.tracks.map((player) => {
        const px = player.x[frame];
        const py = player.y[frame];
        if (isNaN(px) || isNaN(py)) return null;

        const dist = Math.sqrt((px - bx) ** 2 + (py - by) ** 2);

        // Draw lane for nearby players (not the ball carrier)
        if (dist > 2 && dist < 20) {
          return (
            <Line
              key={player.trackId}
              points={[
                [bx, 0.1, by],
                [px, 0.1, py],
              ]}
              color="#00f2ff"
              lineWidth={2}
              transparent
              opacity={0.5}
            />
          );
        }
        return null;
      })}
    </group>
  );
};
