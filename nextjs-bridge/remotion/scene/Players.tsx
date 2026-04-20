import * as THREE from "three";
import { useMemo, useRef, useState, useEffect, Suspense } from "react";
import { useFrame } from "@react-three/fiber";
import { staticFile } from "remotion";
import type { ReplayTable, TrackTrajectory } from "../data/types";
import { PLAYER_HEIGHT, PLAYER_RADIUS, pitchToWorld } from "./constants";
import { PlayerRig } from "./PlayerRig";

type Props = {
  table: ReplayTable;
  frame: number;
  videoFps: number;
  /** When true, PlayerRigs teleport to position instead of smooth transition. */
  teleport?: boolean;
};

/**
 * IDENTITY SYNC: Each track keyed by trackId.
 * trackId = player_id from the video detection pipeline.
 * If video says Player 7 is sprinting, the PlayerRig with key=7 gets that data.
 */
export const Players: React.FC<Props> = ({ table, frame, teleport }) => {
  const [glbOk, setGlbOk] = useState<boolean | null>(null);

  useEffect(() => {
    fetch(staticFile("/avatars/player.glb"))
      .then((r) => setGlbOk(r.ok))
      .catch(() => setGlbOk(false));
  }, []);

  if (glbOk === true) {
    return (
      <Suspense fallback={<CapsuleFallback tracks={table.tracks} table={table} frame={frame} />}>
        {table.tracks.map((t) => (
          <PlayerRig
            key={t.trackId}
            track={t}
            ball={table.ball}
            teamColor={t.teamId === 0 ? table.team0Color : t.teamId === 1 ? table.team1Color : "#888888"}
            frame={frame}
            teleport={teleport}
          />
        ))}
      </Suspense>
    );
  }

  // Capsule fallback
  const team0 = table.tracks.filter((t) => t.teamId === 0);
  const team1 = table.tracks.filter((t) => t.teamId === 1);
  const unknown = table.tracks.filter((t) => t.teamId !== 0 && t.teamId !== 1);
  return (
    <>
      <TeamInstances tracks={team0} color={table.team0Color} frame={frame} />
      <TeamInstances tracks={team1} color={table.team1Color} frame={frame} />
      <TeamInstances tracks={unknown} color="#888888" frame={frame} />
    </>
  );
};

type CapsuleProps = { tracks: TrackTrajectory[]; table: ReplayTable; frame: number };

const CapsuleFallback: React.FC<CapsuleProps> = ({ tracks, table, frame }) => {
  const team0 = useMemo(() => tracks.filter((t) => t.teamId === 0), [tracks]);
  const team1 = useMemo(() => tracks.filter((t) => t.teamId === 1), [tracks]);
  const unk = useMemo(() => tracks.filter((t) => t.teamId !== 0 && t.teamId !== 1), [tracks]);
  return (
    <>
      <TeamInstances tracks={team0} color={table.team0Color} frame={frame} />
      <TeamInstances tracks={team1} color={table.team1Color} frame={frame} />
      <TeamInstances tracks={unk} color="#888888" frame={frame} />
    </>
  );
};

type TeamProps = { tracks: TrackTrajectory[]; color: string; frame: number };
const HIDDEN_Y = -1000;

const TeamInstances: React.FC<TeamProps> = ({ tracks, color, frame }) => {
  const ref = useRef<THREE.InstancedMesh>(null!);
  const tmp = useMemo(() => new THREE.Object3D(), []);
  const count = Math.max(tracks.length, 1);

  useFrame(() => {
    if (!ref.current) return;
    for (let i = 0; i < tracks.length; i++) {
      const trk = tracks[i];
      // Direct index — no interpolation
      const x = trk.x[frame];
      const y = trk.y[frame];
      if (x === undefined || y === undefined || Number.isNaN(x) || Number.isNaN(y)) {
        tmp.position.set(0, HIDDEN_Y, 0);
        tmp.scale.set(0, 0, 0);
      } else {
        const [wx, wz] = pitchToWorld(x, y);
        tmp.position.set(wx, PLAYER_HEIGHT / 2, wz);
        tmp.scale.set(1, 1, 1);
      }
      tmp.updateMatrix();
      ref.current.setMatrixAt(i, tmp.matrix);
    }
    ref.current.instanceMatrix.needsUpdate = true;
  });

  return (
    <instancedMesh ref={ref} args={[undefined, undefined, count]} castShadow receiveShadow>
      <capsuleGeometry args={[PLAYER_RADIUS, PLAYER_HEIGHT - 2 * PLAYER_RADIUS, 4, 12]} />
      <meshStandardMaterial color={color} roughness={0.55} metalness={0.05} />
    </instancedMesh>
  );
};
