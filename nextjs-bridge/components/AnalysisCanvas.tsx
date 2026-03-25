"use client";

import { useRef, useEffect } from "react";

export default function AnalysisCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const W = 1200, H = 700;
    canvas.width = W;
    canvas.height = H;

    // Players: team A (teal), team B (white)
    const teamA = [
      { x: 100, y: 350, id: 1, spd: "12.4", role: "GK" },
      { x: 250, y: 180, id: 2, spd: "24.1", role: "CB" },
      { x: 250, y: 310, id: 5, spd: "22.8", role: "CB" },
      { x: 250, y: 440, id: 3, spd: "26.3", role: "CB" },
      { x: 250, y: 550, id: 6, spd: "25.7", role: "LB" },
      { x: 420, y: 250, id: 8, spd: "28.9", role: "CM" },
      { x: 420, y: 450, id: 4, spd: "27.1", role: "CM" },
      { x: 500, y: 350, id: 10, spd: "30.2", role: "CAM" },
      { x: 620, y: 220, id: 7, spd: "31.5", role: "RW" },
      { x: 620, y: 480, id: 11, spd: "32.1", role: "LW" },
      { x: 650, y: 350, id: 9, spd: "29.4", role: "ST" },
    ];

    const teamB = [
      { x: 1100, y: 350, id: 1, spd: "11.8", role: "GK" },
      { x: 950, y: 200, id: 4, spd: "23.5", role: "CB" },
      { x: 950, y: 350, id: 5, spd: "22.1", role: "CB" },
      { x: 950, y: 500, id: 6, spd: "24.8", role: "CB" },
      { x: 800, y: 150, id: 2, spd: "26.9", role: "RB" },
      { x: 800, y: 550, id: 3, spd: "25.4", role: "LB" },
      { x: 780, y: 280, id: 8, spd: "27.6", role: "CM" },
      { x: 780, y: 420, id: 7, spd: "28.3", role: "CM" },
      { x: 680, y: 350, id: 10, spd: "30.8", role: "CAM" },
      { x: 600, y: 250, id: 9, spd: "31.2", role: "ST" },
      { x: 600, y: 450, id: 11, spd: "29.7", role: "LW" },
    ];

    const ball = { x: 520, y: 340 };
    const offsets = [...teamA, ...teamB].map(() => ({
      ax: (Math.random() - 0.5) * 30,
      ay: (Math.random() - 0.5) * 20,
      phase: Math.random() * Math.PI * 2,
      freq: 0.3 + Math.random() * 0.4,
    }));
    const ballOffset = { ax: 80, ay: 50, phase: 0, freq: 0.15 };
    const ballTrail: { x: number; y: number }[] = [];

    const hudLines = [
      "YOLO v8 — 22 detections",
      "BoT-SORT — 22 active tracks",
      "Team separation — K-means converged",
      "Gemini — analysing tactical shape",
      "Claude — auditing player metrics",
      "Formation: 4-3-3 vs 4-2-3-1",
      "Possession: TEAL 58% — WHITE 42%",
      "Press intensity: HIGH",
    ];

    let frameId: number;
    let t = 0;

    const drawPitch = () => {
      const stripeW = W / 20;
      for (let i = 0; i < 20; i++) {
        ctx.fillStyle = i % 2 === 0 ? "#1a4a1a" : "#1d551d";
        ctx.fillRect(i * stripeW, 0, stripeW, H);
      }

      ctx.strokeStyle = "rgba(255,255,255,0.25)";
      ctx.lineWidth = 2;
      ctx.strokeRect(40, 30, W - 80, H - 60);
      ctx.beginPath();
      ctx.moveTo(W / 2, 30);
      ctx.lineTo(W / 2, H - 30);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(W / 2, H / 2, 80, 0, Math.PI * 2);
      ctx.stroke();
      ctx.fillStyle = "rgba(255,255,255,0.4)";
      ctx.beginPath();
      ctx.arc(W / 2, H / 2, 4, 0, Math.PI * 2);
      ctx.fill();

      ctx.strokeRect(40, H / 2 - 140, 140, 280);
      ctx.strokeRect(W - 180, H / 2 - 140, 140, 280);
      ctx.strokeRect(40, H / 2 - 70, 60, 140);
      ctx.strokeRect(W - 100, H / 2 - 70, 60, 140);
      ctx.fillStyle = "rgba(255,255,255,0.3)";
      ctx.beginPath();
      ctx.arc(150, H / 2, 3, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.arc(W - 150, H / 2, 3, 0, Math.PI * 2);
      ctx.fill();

      for (const [cx, cy] of [[40, 30], [W - 40, 30], [40, H - 30], [W - 40, H - 30]]) {
        ctx.beginPath();
        const startA = cx < W / 2 ? (cy < H / 2 ? 0 : -Math.PI / 2) : (cy < H / 2 ? Math.PI / 2 : Math.PI);
        ctx.arc(cx, cy, 20, startA, startA + Math.PI / 2);
        ctx.stroke();
      }
    };

    const drawPlayer = (
      x: number, y: number, color: string, glow: string, id: number, spd: string, showBox: boolean
    ) => {
      if (showBox) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 3]);
        ctx.strokeRect(x - 18, y - 40, 36, 55);
        ctx.setLineDash([]);
        ctx.fillStyle = color;
        ctx.font = "bold 10px monospace";
        ctx.fillText(`#${id}`, x - 10, y - 44);
        ctx.fillStyle = "rgba(255,255,255,0.5)";
        ctx.font = "9px monospace";
        ctx.fillText(`${spd} km/h`, x - 16, y + 24);
      }

      ctx.shadowColor = glow;
      ctx.shadowBlur = 15;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, 8, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.fillStyle = "rgba(255,255,255,0.6)";
      ctx.beginPath();
      ctx.arc(x, y - 1, 3, 0, Math.PI * 2);
      ctx.fill();
    };

    const drawBall = (x: number, y: number) => {
      for (let i = 0; i < ballTrail.length; i++) {
        const alpha = (i / ballTrail.length) * 0.3;
        ctx.fillStyle = `rgba(255,255,255,${alpha})`;
        ctx.beginPath();
        ctx.arc(ballTrail[i].x, ballTrail[i].y, 2, 0, Math.PI * 2);
        ctx.fill();
      }

      ctx.shadowColor = "#ffffff";
      ctx.shadowBlur = 12;
      ctx.fillStyle = "#ffffff";
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.strokeStyle = "rgba(0,0,0,0.3)";
      ctx.lineWidth = 0.5;
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.stroke();
    };

    const drawHUD = (time: number) => {
      ctx.fillStyle = "rgba(10,15,26,0.85)";
      ctx.fillRect(0, 0, W, 44);
      ctx.fillRect(0, H - 36, W, 36);

      ctx.fillStyle = "#00d4aa";
      ctx.font = "bold 14px monospace";
      ctx.fillText("ATHLINK CV", 16, 18);
      ctx.fillStyle = "rgba(255,255,255,0.5)";
      ctx.font = "11px monospace";
      ctx.fillText("LIVE ANALYSIS", 16, 34);

      const frame = Math.floor(time * 30) % 9000;
      ctx.fillStyle = "rgba(255,255,255,0.4)";
      ctx.font = "11px monospace";
      ctx.textAlign = "right";
      ctx.fillText(`FRAME ${String(frame).padStart(4, "0")}  |  FPS 30`, W - 16, 18);
      ctx.fillText(`22 TRACKS  |  2 TEAMS  |  BALL LOCKED`, W - 16, 34);
      ctx.textAlign = "left";

      const lineIdx = Math.floor(time / 1.5) % hudLines.length;
      const currentLine = hudLines[lineIdx];
      const nextLine = hudLines[(lineIdx + 1) % hudLines.length];
      const blend = (time / 1.5) % 1;

      ctx.fillStyle = `rgba(0,212,170,${1 - blend})`;
      ctx.font = "bold 12px monospace";
      ctx.fillText(`● ${currentLine}`, 16, H - 14);

      if (blend > 0.7) {
        ctx.fillStyle = `rgba(0,212,170,${(blend - 0.7) / 0.3})`;
        ctx.fillText(`● ${nextLine}`, 16, H - 14);
      }

      const pipelineProgress = (time % 8) / 8;
      ctx.fillStyle = "rgba(255,255,255,0.1)";
      ctx.fillRect(W - 220, H - 22, 200, 6);
      ctx.fillStyle = "#00d4aa";
      ctx.fillRect(W - 220, H - 22, 200 * pipelineProgress, 6);
      ctx.fillStyle = "rgba(255,255,255,0.4)";
      ctx.font = "9px monospace";
      ctx.textAlign = "right";
      ctx.fillText(`PIPELINE ${Math.floor(pipelineProgress * 100)}%`, W - 16, H - 12);
      ctx.textAlign = "left";

      ctx.strokeStyle = "rgba(0,212,170,0.12)";
      ctx.lineWidth = 1;
      const defs = teamA.slice(1, 5);
      ctx.beginPath();
      for (let i = 0; i < defs.length; i++) {
        const off = offsets[1 + i];
        const px = defs[i].x + Math.sin(time * off.freq + off.phase) * off.ax;
        const py = defs[i].y + Math.cos(time * off.freq + off.phase) * off.ay;
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.stroke();
      const mids = teamA.slice(5, 8);
      ctx.beginPath();
      for (let i = 0; i < mids.length; i++) {
        const off = offsets[5 + i];
        const px = mids[i].x + Math.sin(time * off.freq + off.phase) * off.ax;
        const py = mids[i].y + Math.cos(time * off.freq + off.phase) * off.ay;
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.stroke();
    };

    const drawScanlines = () => {
      ctx.fillStyle = "rgba(0,0,0,0.03)";
      for (let y = 0; y < H; y += 4) {
        ctx.fillRect(0, y, W, 1);
      }
    };

    const animate = () => {
      frameId = requestAnimationFrame(animate);
      t += 1 / 60;
      ctx.clearRect(0, 0, W, H);

      drawPitch();

      const allPlayers = [...teamA, ...teamB];
      const showBoxPhase = Math.floor(t * 2) % 4;

      allPlayers.forEach((p, i) => {
        const off = offsets[i];
        const px = p.x + Math.sin(t * off.freq + off.phase) * off.ax;
        const py = p.y + Math.cos(t * off.freq + off.phase) * off.ay;
        const isTeamA = i < teamA.length;
        const color = isTeamA ? "#00d4aa" : "rgba(255,255,255,0.9)";
        const glow = isTeamA ? "rgba(0,212,170,0.6)" : "rgba(255,255,255,0.4)";
        const showBox = (i % 4) === showBoxPhase || (Math.sin(t + i) > 0.3);
        drawPlayer(px, py, color, glow, p.id, p.spd, showBox);
      });

      const bx = ball.x + Math.sin(t * ballOffset.freq) * ballOffset.ax;
      const by = ball.y + Math.cos(t * ballOffset.freq * 1.3) * ballOffset.ay;
      ballTrail.push({ x: bx, y: by });
      if (ballTrail.length > 20) ballTrail.shift();
      drawBall(bx, by);

      drawHUD(t);
      drawScanlines();
    };

    animate();
    return () => cancelAnimationFrame(frameId);
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="w-full h-full"
      style={{ display: "block", borderRadius: "12px" }}
    />
  );
}
