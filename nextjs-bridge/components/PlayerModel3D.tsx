"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import type { FootballPlayer3DProps } from "@/components/FootballPlayer3D";

const FootballPlayer3D = dynamic(() => import("@/components/FootballPlayer3D"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full flex items-center justify-center">
      <div className="flex flex-col items-center gap-3">
        <div className="w-8 h-8 border-2 border-[#00d4aa] border-t-transparent rounded-full animate-spin" />
        <span className="text-[11px] text-gray-500 font-mono tracking-wider">LOADING MODEL</span>
      </div>
    </div>
  ),
}) as React.ComponentType<FootballPlayer3DProps>;

const JERSEY_PRESETS = [
  { name: "Teal",   body: "#00d4aa" },
  { name: "Red",    body: "#dc2626" },
  { name: "Blue",   body: "#2563eb" },
  { name: "Yellow", body: "#eab308" },
  { name: "White",  body: "#e8e8e8" },
  { name: "Black",  body: "#2a2a2a" },
  { name: "Orange", body: "#ea580c" },
  { name: "Purple", body: "#7c3aed" },
];

const SKIN_TONES = [
  { name: "Light",  color: "#f5d0a9" },
  { name: "Medium", color: "#c68642" },
  { name: "Tan",    color: "#a0724a" },
  { name: "Brown",  color: "#6b4226" },
  { name: "Dark",   color: "#3b2114" },
];

const HAIR_COLORS = [
  { name: "Black",  color: "#1a1a1a" },
  { name: "Brown",  color: "#4a2c0a" },
  { name: "Blonde", color: "#c9a435" },
  { name: "Red",    color: "#8b2500" },
  { name: "White",  color: "#d4d4d4" },
];

interface PlayerModel3DProps {
  playerName?: string;
  playerNumber?: number;
}

export default function PlayerModel3D({
  playerName = "PLAYER",
  playerNumber = 10,
}: PlayerModel3DProps) {
  const [jerseyPreset, setJerseyPreset] = useState(0);
  const [skinTone, setSkinTone] = useState(1);
  const [hairColor, setHairColor] = useState(0);

  const kitColour = JERSEY_PRESETS[jerseyPreset].body;
  const skinColour = SKIN_TONES[skinTone].color;
  const hairColour = HAIR_COLORS[hairColor].color;

  return (
    <div className="relative w-full h-full flex">
      {/* 3D Canvas */}
      <div
        className="flex-1 min-h-[500px] relative"
        style={{
          background:
            "radial-gradient(ellipse at 50% 85%, rgba(0,212,170,0.10) 0%, #080c14 70%)",
        }}
      >
        <FootballPlayer3D
          kitColour={kitColour}
          skinColour={skinColour}
          hairColour={hairColour}
        />
      </div>

      {/* Customise Panel */}
      <div className="w-52 flex-shrink-0 bg-[#0a0e16] border-l border-white/5 p-4 overflow-y-auto">
        <p className="text-[10px] text-gray-600 uppercase tracking-widest mb-4">Customise</p>

        {/* Kit */}
        <div className="mb-5">
          <p className="text-[11px] text-gray-400 mb-2 font-semibold">Kit</p>
          <div className="grid grid-cols-4 gap-1.5">
            {JERSEY_PRESETS.map((p, i) => (
              <button
                key={p.name}
                onClick={() => setJerseyPreset(i)}
                className={`w-8 h-8 rounded-md border-2 transition-all ${
                  i === jerseyPreset
                    ? "border-[#00d4aa] scale-110"
                    : "border-transparent hover:border-white/20"
                }`}
                style={{ background: p.body }}
                title={p.name}
              />
            ))}
          </div>
        </div>

        {/* Skin */}
        <div className="mb-5">
          <p className="text-[11px] text-gray-400 mb-2 font-semibold">Skin</p>
          <div className="flex gap-1.5">
            {SKIN_TONES.map((s, i) => (
              <button
                key={s.name}
                onClick={() => setSkinTone(i)}
                className={`w-8 h-8 rounded-full border-2 transition-all ${
                  i === skinTone
                    ? "border-[#00d4aa] scale-110"
                    : "border-transparent hover:border-white/20"
                }`}
                style={{ background: s.color }}
                title={s.name}
              />
            ))}
          </div>
        </div>

        {/* Hair */}
        <div className="mb-5">
          <p className="text-[11px] text-gray-400 mb-2 font-semibold">Hair</p>
          <div className="flex gap-1.5">
            {HAIR_COLORS.map((h, i) => (
              <button
                key={h.name}
                onClick={() => setHairColor(i)}
                className={`w-8 h-8 rounded-full border-2 transition-all ${
                  i === hairColor
                    ? "border-[#00d4aa] scale-110"
                    : "border-transparent hover:border-white/20"
                }`}
                style={{ background: h.color }}
                title={h.name}
              />
            ))}
          </div>
        </div>

        {/* Player badge */}
        <div className="mt-3 pt-3 border-t border-white/5">
          <div className="flex items-center gap-2">
            <span className="text-2xl font-bold text-[#00d4aa] font-mono">{playerNumber}</span>
            <span className="text-[11px] text-gray-400 uppercase tracking-wider">{playerName}</span>
          </div>
        </div>

        <div className="mt-3 pt-3 border-t border-white/5">
          <p className="text-[9px] text-gray-600 leading-relaxed">
            Drag to rotate. Scroll to zoom.
          </p>
        </div>
      </div>
    </div>
  );
}
