"use client";

import { useRef, useCallback, useEffect, useState } from "react";
import { motion, useInView } from "framer-motion";
import dynamic from "next/dynamic";
import { Upload, Filter, Eye, Users, Layers, Brain, ShieldCheck, FileText, X } from "lucide-react";
import { GlowingEffect } from "@/components/ui/glowing-effect";
import AnalysisCanvas from "@/components/AnalysisCanvas";

const PlayerModel3D = dynamic(() => import("@/components/PlayerModel3D"), { ssr: false });

const APP_URL = "https://web-production-6b7ca.up.railway.app";


/* ============================================================
   FEATURE CARD WITH 3D TILT + GLOWING EFFECT
   ============================================================ */
function FeatureCard({
  title,
  description,
  icon,
}: {
  title: string;
  description: string;
  icon: React.ReactNode;
}) {
  const cardRef = useRef<HTMLDivElement>(null);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const el = cardRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;
    const rotateX = ((y - centerY) / centerY) * -8;
    const rotateY = ((x - centerX) / centerX) * 8;
    el.style.transform = `perspective(600px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-4px)`;
  }, []);

  const handleMouseLeave = useCallback(() => {
    const el = cardRef.current;
    if (!el) return;
    el.style.transform = "perspective(600px) rotateX(0) rotateY(0) translateY(0)";
  }, []);

  return (
    <div
      ref={cardRef}
      className="feature-card relative rounded-xl bg-[#111827] p-6 border border-white/5 overflow-hidden"
      style={{ transformStyle: "preserve-3d", transition: "transform 0.3s ease" }}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
    >
      <GlowingEffect
        spread={40}
        glow
        disabled={false}
        proximity={64}
        inactiveZone={0.01}
        borderWidth={2}
      />
      <div className="relative z-10">
        <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-[#00d4aa]/20 to-[#00a88a]/10 flex items-center justify-center mb-4 text-[#00d4aa]">
          {icon}
        </div>
        <h3 className="text-xl font-bold text-white mb-2">{title}</h3>
        <p className="text-gray-400 text-sm leading-relaxed">{description}</p>
      </div>
    </div>
  );
}

/* ============================================================
   FADE-IN ANIMATION WRAPPER
   ============================================================ */
function FadeInSection({
  children,
  className = "",
  delay = 0,
}: {
  children: React.ReactNode;
  className?: string;
  delay?: number;
}) {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-80px" });

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 40 }}
      animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 40 }}
      transition={{ duration: 0.7, delay, ease: [0.25, 0.1, 0.25, 1] }}
      className={className}
    >
      {children}
    </motion.div>
  );
}

/* ============================================================
   ICONS (inline SVG)
   ============================================================ */
const IconReport = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
    <polyline points="14 2 14 8 20 8" />
    <line x1="16" y1="13" x2="8" y2="13" />
    <line x1="16" y1="17" x2="8" y2="17" />
    <polyline points="10 9 9 9 8 9" />
  </svg>
);

const IconTracking = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10" />
    <circle cx="12" cy="12" r="6" />
    <circle cx="12" cy="12" r="2" />
  </svg>
);

const IconTeam = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
    <circle cx="9" cy="7" r="4" />
    <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
    <path d="M16 3.13a4 4 0 0 1 0 7.75" />
  </svg>
);


/* ============================================================
   PIPELINE DIAGRAM
   ============================================================ */
const PIPELINE_NODES = [
  {
    id: 1,
    label: "Video Upload",
    sub: "Drop in any match clip — phone footage, broadcast, or dashcam. Any format works.",
    icon: Upload,
    glow: "",
  },
  {
    id: 2,
    label: "Scene Filter",
    sub: "Crowd shots, bench cuts, and replays are automatically removed. Only live pitch action counts.",
    icon: Filter,
    glow: "",
  },
  {
    id: 3,
    label: "Player Detection",
    sub: "Every player on the pitch is located in under 60 milliseconds per frame using AI vision.",
    icon: Eye,
    glow: "",
  },
  {
    id: 4,
    label: "Player Tracking",
    sub: "Each player receives a unique ID and is followed continuously across every frame of the match.",
    icon: Users,
    glow: "",
  },
  {
    id: 5,
    label: "Team Separation",
    sub: "Players are sorted into two teams automatically based on their kit colour. No tagging needed.",
    icon: Layers,
    glow: "",
  },
  {
    id: 6,
    label: "Gemini Vision",
    sub: "Google's AI watches your footage and identifies tactical patterns, pressure zones, and key moments.",
    icon: Brain,
    glow: "0 0 20px rgba(99,102,241,0.3)",
  },
  {
    id: 7,
    label: "Accuracy Check",
    sub: "A second AI reviews all the numbers and removes anything that looks incorrect or unreliable.",
    icon: ShieldCheck,
    glow: "0 0 20px rgba(251,146,60,0.3)",
  },
  {
    id: 8,
    label: "Coaching Report",
    sub: "You get a full report — player stats, team shape, sprint data, and specific training recommendations.",
    icon: FileText,
    glow: "0 0 24px rgba(0,212,170,0.4)",
  },
];

/* ============================================================
   STEP ANIMATIONS
   ============================================================ */
function StepAnimation({ stepId }: { stepId: number }) {
  if (stepId === 1) return (
    <div className="w-full h-40 flex items-center justify-center gap-8 relative overflow-hidden">
      <style>{`
        @keyframes slideInLeft { from { opacity:0; transform:translateX(-40px); } to { opacity:1; transform:translateX(0); } }
        @keyframes slideInRight { from { opacity:0; transform:translateX(40px); } to { opacity:1; transform:translateX(0); } }
        @keyframes bounceUp { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-12px)} }
      `}</style>
      <div style={{ animation: "slideInLeft 0.6s ease forwards" }}>
        <svg width="48" height="60" viewBox="0 0 48 60"><rect x="4" y="4" width="40" height="52" rx="4" fill="none" stroke="#00d4aa" strokeWidth="2"/><line x1="12" y1="16" x2="36" y2="16" stroke="#00d4aa" strokeWidth="1.5"/><line x1="12" y1="24" x2="36" y2="24" stroke="#00d4aa" strokeWidth="1.5"/><line x1="12" y1="32" x2="28" y2="32" stroke="#00d4aa" strokeWidth="1.5"/></svg>
      </div>
      <div style={{ animation: "bounceUp 1.2s ease infinite" }}>
        <svg width="40" height="40" viewBox="0 0 40 40"><polyline points="20,32 20,8" stroke="white" strokeWidth="2.5" strokeLinecap="round"/><polyline points="10,18 20,8 30,18" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" fill="none"/><line x1="8" y1="34" x2="32" y2="34" stroke="#00d4aa" strokeWidth="2" strokeLinecap="round"/></svg>
      </div>
      <div style={{ animation: "slideInRight 0.6s ease 0.2s both" }}>
        <svg width="48" height="60" viewBox="0 0 48 60"><rect x="4" y="4" width="40" height="52" rx="4" fill="none" stroke="#00d4aa" strokeWidth="2"/><line x1="12" y1="16" x2="36" y2="16" stroke="#00d4aa" strokeWidth="1.5"/><line x1="12" y1="24" x2="36" y2="24" stroke="#00d4aa" strokeWidth="1.5"/><line x1="12" y1="32" x2="28" y2="32" stroke="#00d4aa" strokeWidth="1.5"/></svg>
      </div>
    </div>
  );

  if (stepId === 2) return (
    <div className="w-full h-40 flex items-center justify-center gap-12">
      <style>{`
        @keyframes crossIn { from{opacity:0;transform:scale(0.5)rotate(-45deg)}to{opacity:1;transform:scale(1)rotate(0deg)} }
        @keyframes checkIn { from{opacity:0;transform:scale(0.5)}to{opacity:1;transform:scale(1)} }
      `}</style>
      <div className="flex flex-col items-center gap-2">
        <svg width="56" height="40" viewBox="0 0 56 40"><rect x="2" y="2" width="52" height="36" rx="4" fill="#1a1a2e" stroke="#444" strokeWidth="1.5"/><circle cx="14" cy="28" r="6" fill="#333"/><circle cx="22" cy="28" r="8" fill="#333"/><circle cx="34" cy="28" r="8" fill="#333"/><circle cx="42" cy="28" r="6" fill="#333"/><ellipse cx="28" cy="20" rx="10" ry="4" fill="#333"/></svg>
        <div style={{ animation: "crossIn 0.5s ease forwards" }}>
          <svg width="32" height="32" viewBox="0 0 32 32"><line x1="4" y1="4" x2="28" y2="28" stroke="#ef4444" strokeWidth="3" strokeLinecap="round"/><line x1="28" y1="4" x2="4" y2="28" stroke="#ef4444" strokeWidth="3" strokeLinecap="round"/></svg>
        </div>
      </div>
      <div className="flex flex-col items-center gap-2">
        <svg width="56" height="36" viewBox="0 0 56 36"><rect x="2" y="2" width="52" height="32" rx="4" fill="#0a2a1a" stroke="#00d4aa" strokeWidth="1.5"/><ellipse cx="28" cy="20" rx="18" ry="8" fill="#0d4a2a"/></svg>
        <div style={{ animation: "checkIn 0.5s ease 0.3s both" }}>
          <svg width="32" height="32" viewBox="0 0 32 32"><polyline points="4,18 12,26 28,8" stroke="#22c55e" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" fill="none"/></svg>
        </div>
      </div>
    </div>
  );

  if (stepId === 3) return (
    <div className="w-full h-40 flex items-center justify-center">
      <style>{`
        @keyframes dotAppear { from{opacity:0;transform:scale(0)} 60%{transform:scale(1.4)} to{opacity:1;transform:scale(1)} }
      `}</style>
      <svg width="160" height="100" viewBox="0 0 160 100">
        <rect x="2" y="2" width="156" height="96" rx="6" fill="#0a2a1a" stroke="#00d4aa" strokeWidth="1.5"/>
        {[
          [30,30],[60,20],[90,35],[120,25],[45,60],[80,65],[110,55],[140,40],[25,75],[65,80],[100,70]
        ].map(([cx,cy],i) => (
          <circle key={i} cx={cx} cy={cy} r="5" fill="#00d4aa" style={{ animation: `dotAppear 0.4s ease ${i*0.12}s both` }}/>
        ))}
      </svg>
    </div>
  );

  if (stepId === 4) return (
    <div className="w-full h-40 flex items-center justify-center">
      <style>{`
        @keyframes moveDot1 { 0%{offset-distance:0%}100%{offset-distance:100%} }
        @keyframes moveDot2 { 0%{offset-distance:0%}100%{offset-distance:100%} }
        @keyframes trailFade { 0%,100%{opacity:0}50%{opacity:0.5} }
      `}</style>
      <svg width="160" height="100" viewBox="0 0 160 100">
        <rect x="2" y="2" width="156" height="96" rx="6" fill="#0a2a1a" stroke="#00d4aa" strokeWidth="1.5"/>
        <path d="M20,70 Q50,20 90,50 T150,30" fill="none" stroke="#00d4aa" strokeWidth="1" strokeDasharray="4 3" opacity="0.4"/>
        <path d="M20,50 Q60,80 100,40 T150,60" fill="none" stroke="#00d4aa" strokeWidth="1" strokeDasharray="4 3" opacity="0.4"/>
        <path d="M30,80 Q80,30 130,70" fill="none" stroke="#00d4aa" strokeWidth="1" strokeDasharray="4 3" opacity="0.4"/>
        <circle r="6" fill="#ef4444" style={{ offsetPath: "path('M20,70 Q50,20 90,50 T150,30')", animation: "moveDot1 2s linear infinite" } as React.CSSProperties}/>
        <circle r="6" fill="#3b82f6" style={{ offsetPath: "path('M20,50 Q60,80 100,40 T150,60')", animation: "moveDot2 2.5s linear infinite 0.5s" } as React.CSSProperties}/>
        <circle r="6" fill="#00d4aa" style={{ offsetPath: "path('M30,80 Q80,30 130,70')", animation: "moveDot1 1.8s linear infinite 1s" } as React.CSSProperties}/>
      </svg>
    </div>
  );

  if (stepId === 5) return (
    <div className="w-full h-40 flex items-center justify-center">
      <style>{`
        @keyframes toRed { 0%{fill:#888}100%{fill:#ef4444} }
        @keyframes toBlue { 0%{fill:#888}100%{fill:#3b82f6} }
      `}</style>
      <svg width="160" height="100" viewBox="0 0 160 100">
        <rect x="2" y="2" width="156" height="96" rx="6" fill="#0a2a1a" stroke="#00d4aa" strokeWidth="1.5"/>
        {[
          [40,30,"red"],[60,25,"red"],[35,55,"red"],[55,60,"red"],[48,42,"red"],
          [100,30,"blue"],[120,25,"blue"],[105,55,"blue"],[125,60,"blue"],[115,42,"blue"],
        ].map(([cx,cy,team],i) => (
          <circle key={i} cx={cx as number} cy={cy as number} r="7"
            style={{ animation: `${team === "red" ? "toRed" : "toBlue"} 0.8s ease ${0.5 + i*0.1}s both` }}/>
        ))}
        <line x1="80" y1="10" x2="80" y2="90" stroke="#00d4aa" strokeWidth="1" strokeDasharray="4 3" opacity="0.5"/>
      </svg>
    </div>
  );

  if (stepId === 6) return (
    <div className="w-full h-40 flex items-center justify-center">
      <style>{`
        @keyframes scanLine { 0%{transform:translateX(-60px)}100%{transform:translateX(60px)} }
      `}</style>
      <div className="relative">
        <svg width="140" height="100" viewBox="0 0 140 100">
          <rect x="10" y="10" width="120" height="80" rx="6" fill="#111" stroke="#444" strokeWidth="1.5"/>
          <rect x="20" y="20" width="100" height="60" rx="4" fill="#0a2a1a"/>
          <ellipse cx="70" cy="50" rx="25" ry="15" fill="none" stroke="#00d4aa" strokeWidth="2"/>
          <circle cx="70" cy="50" r="8" fill="none" stroke="#00d4aa" strokeWidth="1.5"/>
          <circle cx="70" cy="50" r="3" fill="#00d4aa"/>
          <rect x="20" y="20" width="4" height="60" rx="2" fill="#00d4aa" opacity="0.7"
            style={{ animation: "scanLine 1.5s ease-in-out infinite" }}/>
        </svg>
      </div>
    </div>
  );

  if (stepId === 7) return (
    <div className="w-full h-40 flex items-center justify-center">
      <style>{`
        @keyframes itemAppear { from{opacity:0;transform:translateX(-10px)}to{opacity:1;transform:translateX(0)} }
        @keyframes tickAppear { from{opacity:0;transform:scale(0)}to{opacity:1;transform:scale(1)} }
      `}</style>
      <div className="flex flex-col gap-3">
        {["Sprint data validated","Team shape confirmed","Possession verified","Distances cross-checked"].map((item,i) => (
          <div key={i} className="flex items-center gap-3" style={{ animation: `itemAppear 0.4s ease ${i*0.4}s both` }}>
            <div style={{ animation: `tickAppear 0.3s ease ${i*0.4+0.3}s both` }}>
              <svg width="20" height="20" viewBox="0 0 20 20"><circle cx="10" cy="10" r="9" fill="#22c55e" opacity="0.2"/><polyline points="4,10 8,14 16,6" stroke="#22c55e" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" fill="none"/></svg>
            </div>
            <span className="text-white text-sm">{item}</span>
          </div>
        ))}
      </div>
    </div>
  );

  if (stepId === 8) return (
    <div className="w-full h-40 flex items-center justify-center">
      <style>{`
        @keyframes lineType { from{width:0}to{width:100%} }
        @keyframes unfold { from{opacity:0;transform:scaleY(0)}to{opacity:1;transform:scaleY(1)} }
      `}</style>
      <div className="flex items-start gap-4">
        <svg width="48" height="60" viewBox="0 0 48 60" style={{ animation: "unfold 0.5s ease forwards", transformOrigin: "top" }}>
          <rect x="4" y="4" width="40" height="52" rx="4" fill="#0a2a1a" stroke="#00d4aa" strokeWidth="2"/>
          <path d="M28 4 L28 16 L40 16" fill="none" stroke="#00d4aa" strokeWidth="1.5"/>
          <rect x="4" y="4" width="24" height="12" fill="#0a2a1a"/>
        </svg>
        <div className="flex flex-col gap-2 pt-1">
          {[80,65,90,55,70].map((w,i) => (
            <div key={i} className="h-2 bg-[#00d4aa]/30 rounded overflow-hidden" style={{ width: "100px" }}>
              <div className="h-full bg-[#00d4aa] rounded" style={{ width: `${w}%`, animation: `lineType 0.5s ease ${0.3+i*0.2}s both` }}/>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  return null;
}

/* ============================================================
   PIPELINE MODAL
   ============================================================ */
function PipelineModal({ node, onClose }: { node: typeof PIPELINE_NODES[0]; onClose: () => void }) {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);

  const Icon = node.icon;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80"
      onClick={onClose}
    >
      <div
        className="relative w-full max-w-lg rounded-2xl border border-white/10 bg-[#0a0e1a] p-8 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-gray-400 hover:text-white transition-colors"
        >
          <X size={20} />
        </button>

        <div className="flex items-center gap-3 mb-2">
          <div className="w-10 h-10 rounded-lg bg-[#00d4aa]/10 flex items-center justify-center text-[#00d4aa]">
            <Icon size={20} />
          </div>
          <p className="text-[#00d4aa] text-xs font-semibold uppercase tracking-widest">
            Step {node.id} of 8
          </p>
        </div>

        <h3 className="text-2xl font-bold text-white mb-4">{node.label}</h3>
        <p className="text-gray-400 leading-relaxed mb-6">{node.sub}</p>

        <div className="rounded-xl bg-[#060a14] border border-white/5 overflow-hidden">
          <StepAnimation stepId={node.id} />
        </div>
      </div>
    </div>
  );
}

function PipelineDiagram({ onStepClick }: { onStepClick: (node: typeof PIPELINE_NODES[0]) => void }) {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-60px" });

  return (
    <div ref={ref}>
      {/* Desktop: horizontal */}
      <div className="hidden md:flex items-start justify-center gap-0">
        {PIPELINE_NODES.map((node, i) => {
          const Icon = node.icon;
          return (
            <div key={node.id} className="flex items-center">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
                transition={{ duration: 0.5, delay: i * 0.15, ease: [0.25, 0.1, 0.25, 1] }}
                className="flex flex-col items-center text-center w-[110px] cursor-pointer group"
                onClick={() => onStepClick(node)}
              >
                <div
                  className="w-14 h-14 rounded-xl border border-[rgba(0,212,170,0.4)] bg-[#0d1117] flex items-center justify-center mb-3 group-hover:border-[#00d4aa] group-hover:bg-[#0d1f1a] transition-colors"
                  style={node.glow ? { boxShadow: node.glow } : undefined}
                >
                  <Icon size={22} className="text-[#00d4aa]" />
                </div>
                <span className="text-white text-[11px] font-bold leading-tight mb-1">
                  {node.label}
                </span>
                <span className="text-[#00d4aa] text-[9px] leading-tight opacity-0 group-hover:opacity-100 transition-opacity">
                  tap to learn more
                </span>
              </motion.div>

              {/* Connecting arrow */}
              {i < PIPELINE_NODES.length - 1 && (
                <motion.div
                  initial={{ opacity: 0, scaleX: 0 }}
                  animate={isInView ? { opacity: 1, scaleX: 1 } : { opacity: 0, scaleX: 0 }}
                  transition={{ duration: 0.4, delay: i * 0.15 + 0.1 }}
                  className="flex-shrink-0 mb-8"
                  style={{ originX: 0 }}
                >
                  <svg width="24" height="12" viewBox="0 0 24 12">
                    <line
                      x1="0" y1="6" x2="18" y2="6"
                      stroke="#00d4aa" strokeWidth="1.5"
                      strokeDasharray="4 3"
                      className="pipeline-dash"
                    />
                    <polygon points="18,2 24,6 18,10" fill="#00d4aa" opacity="0.7" />
                  </svg>
                </motion.div>
              )}
            </div>
          );
        })}
      </div>

      {/* Mobile: vertical */}
      <div className="flex md:hidden flex-col items-center gap-0">
        {PIPELINE_NODES.map((node, i) => {
          const Icon = node.icon;
          return (
            <div key={node.id} className="flex flex-col items-center">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
                transition={{ duration: 0.5, delay: i * 0.12, ease: [0.25, 0.1, 0.25, 1] }}
                className="flex items-center gap-4 w-full max-w-xs cursor-pointer"
                onClick={() => onStepClick(node)}
              >
                <div
                  className="w-14 h-14 rounded-xl border border-[rgba(0,212,170,0.4)] bg-[#0d1117] flex-shrink-0 flex items-center justify-center"
                  style={node.glow ? { boxShadow: node.glow } : undefined}
                >
                  <Icon size={20} className="text-[#00d4aa]" />
                </div>
                <div>
                  <p className="text-white text-xs font-bold">{node.label}</p>
                  <p className="text-[#00d4aa] text-[10px] opacity-60">tap to learn more</p>
                </div>
              </motion.div>

              {/* Vertical arrow */}
              {i < PIPELINE_NODES.length - 1 && (
                <motion.div
                  initial={{ opacity: 0, scaleY: 0 }}
                  animate={isInView ? { opacity: 1, scaleY: 1 } : { opacity: 0, scaleY: 0 }}
                  transition={{ duration: 0.3, delay: i * 0.12 + 0.08 }}
                  className="my-1"
                  style={{ originY: 0 }}
                >
                  <svg width="12" height="20" viewBox="0 0 12 20">
                    <line
                      x1="6" y1="0" x2="6" y2="14"
                      stroke="#00d4aa" strokeWidth="1.5"
                      strokeDasharray="4 3"
                      className="pipeline-dash-v"
                    />
                    <polygon points="2,14 6,20 10,14" fill="#00d4aa" opacity="0.7" />
                  </svg>
                </motion.div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ============================================================
   BROADCAST-STYLE TACTICAL ANALYSIS CANVAS
   ============================================================ */

/* ============================================================
   SCROLL EXPAND ANALYSIS
   ============================================================ */
function ScrollExpandAnalysis() {
  const sectionRef = useRef<HTMLDivElement>(null);
  const [progress, setProgress] = useState(0);
  const progressRef = useRef(progress);

  useEffect(() => {
    progressRef.current = progress;
  }, [progress]);

  useEffect(() => {
    const section = sectionRef.current;
    if (!section) return;

    const handleScroll = () => {
      if (!section) return;
      const rect = section.getBoundingClientRect();
      const viewH = window.innerHeight;
      const raw = Math.max(0, Math.min(1, 1 - rect.top / viewH));
      if (Math.abs(raw - progressRef.current) > 0.01) {
        setProgress(raw);
      }
    };

    const observer = new IntersectionObserver(() => handleScroll(), { threshold: [0, 0.5, 1] });
    observer.observe(section);

    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => {
      window.removeEventListener("scroll", handleScroll);
      observer.disconnect();
    };
  }, []);

  const mediaWidth = 300 + progress * 900;
  const mediaHeight = 400 + progress * 300;
  const textTx = progress * 150;
  const contentOpacity = progress >= 0.95 ? 1 : 0;

  return (
    <div ref={sectionRef} className="relative">
      {/* Split text */}
      <div className="flex justify-center gap-4 mb-8">
        <span
          className="text-4xl md:text-6xl font-bold text-white transition-transform duration-100"
          style={{ transform: `translateX(-${textTx}px)` }}
        >
          WATCH
        </span>
        <span
          className="text-4xl md:text-6xl font-bold text-[#00d4aa] transition-transform duration-100"
          style={{ transform: `translateX(${textTx}px)` }}
        >
          IT WORK
        </span>
      </div>

      {/* Expanding media frame */}
      <div className="flex justify-center">
        <div
          className="overflow-hidden rounded-xl border border-[rgba(0,212,170,0.3)] transition-[width,height] duration-100"
          style={{
            width: `min(${mediaWidth}px, 100%)`,
            height: `${mediaHeight}px`,
          }}
        >
          <AnalysisCanvas />
        </div>
      </div>

      {/* Stat cards — appear after full expansion */}
      <div
        className="grid md:grid-cols-3 gap-6 mt-12 max-w-4xl mx-auto transition-opacity duration-500"
        style={{ opacity: contentOpacity }}
      >
        {[
          { title: "Players Detected", sub: "Every visible player tracked per frame" },
          { title: "Teams Separated", sub: "Kit colour clustering, automatic" },
          { title: "Report Generated", sub: "Tactics, players, training plan" },
        ].map((card) => (
          <div
            key={card.title}
            className="rounded-xl bg-[#0d1117] border border-white/5 p-6 text-center"
          >
            <h4 className="text-white font-bold text-lg mb-2">{card.title}</h4>
            <p className="text-gray-500 text-sm">{card.sub}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ============================================================
   MAIN PAGE
   ============================================================ */
export default function Home() {
  const [activeStep, setActiveStep] = useState<typeof PIPELINE_NODES[0] | null>(null);

  return (
    <main className="min-h-screen">
      {activeStep && (
        <PipelineModal node={activeStep} onClose={() => setActiveStep(null)} />
      )}
      {/* ===== NAVBAR ===== */}
      <nav className="fixed top-0 inset-x-0 z-40 bg-[#0a0f1a]/80 backdrop-blur-xl border-b border-white/5">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <span className="text-2xl font-bold tracking-wider text-[#00d4aa]">
            ATHLINK CV
          </span>
          <a href={APP_URL} className="cta-button !py-2 !px-5 !text-sm">
            ANALYSE FOOTAGE →
          </a>
        </div>
      </nav>

      {/* ===== HERO ===== */}
      <section className="relative pt-32 pb-20 px-6 overflow-hidden">
        {/* Background glow */}
        <div className="absolute top-0 left-1/4 w-[600px] h-[600px] bg-[#00d4aa]/5 rounded-full blur-[120px] pointer-events-none" />
        <div className="absolute bottom-0 right-1/4 w-[400px] h-[400px] bg-[#00d4aa]/3 rounded-full blur-[100px] pointer-events-none" />

        <div className="max-w-7xl mx-auto grid lg:grid-cols-2 gap-16 items-center">
          {/* Left — text */}
          <FadeInSection>
            <p className="hero-tagline text-[#00d4aa] text-sm font-semibold mb-4 uppercase">
              Football Analysis For Everyone
            </p>
            <h1 className="hero-headline text-4xl sm:text-5xl lg:text-6xl font-bold text-white leading-tight mb-6">
              Professional Analysis.{" "}
              <span className="text-[#00d4aa]">Not Professional Prices.</span>
            </h1>
            <p className="text-gray-400 text-lg leading-relaxed mb-8 max-w-lg">
              Upload a clip. Get a full coaching report in minutes — player stats, team shape, sprint data, and training recommendations.
            </p>
            <a href={APP_URL} className="cta-button">
              START ANALYSING →
            </a>
          </FadeInSection>

          {/* Right — live analysis preview */}
          <FadeInSection delay={0.3} className="flex justify-center">
            <div className="relative w-full max-w-[560px] aspect-[4/3] rounded-xl overflow-hidden shadow-2xl shadow-black/40">
              <AnalysisCanvas />
              <div className="absolute inset-0 pointer-events-none rounded-xl" style={{ boxShadow: "inset 0 0 60px rgba(0,212,170,0.12)" }} />
            </div>
          </FadeInSection>
        </div>
      </section>

      {/* ===== THE PIPELINE ===== */}
      <section className="py-24 px-6 relative overflow-hidden">
        {/* Subtle teal glow */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[400px] bg-[#00d4aa]/5 rounded-full blur-[160px] pointer-events-none" />

        <div className="max-w-7xl mx-auto relative">
          <FadeInSection>
            <h2 className="section-title">The Pipeline</h2>
            <p className="text-gray-400 text-center mb-16 max-w-xl mx-auto">
              Every job runs through eight layers of analysis. No shortcuts.
            </p>
          </FadeInSection>

          <PipelineDiagram onStepClick={setActiveStep} />
        </div>
      </section>

      {/* ===== SEE THE ANALYSIS ===== */}
      <section className="py-24 px-6 relative overflow-hidden">
        <div className="max-w-7xl mx-auto">
          <FadeInSection>
            <h2 className="section-title mb-16">See The Analysis</h2>
          </FadeInSection>
          <ScrollExpandAnalysis />
        </div>
      </section>

      {/* ===== FEATURES ===== */}
      <section className="py-24 px-6 relative">
        <div className="absolute top-1/2 left-0 w-[500px] h-[500px] bg-[#00d4aa]/3 rounded-full blur-[140px] pointer-events-none -translate-y-1/2" />

        <div className="max-w-5xl mx-auto relative">
          <FadeInSection>
            <h2 className="section-title">Built Different</h2>
            <p className="text-gray-400 text-center mb-16 max-w-xl mx-auto">
              Every feature designed for coaches who need results, not dashboards.
            </p>
          </FadeInSection>

          <div className="grid sm:grid-cols-2 gap-6">
            <FadeInSection delay={0}>
              <FeatureCard
                icon={<IconReport />}
                title="AI Coaching Reports"
                description="Gemini watches the footage. Claude audits the data. You get a report no human analyst could write faster."
              />
            </FadeInSection>
            <FadeInSection delay={0.1}>
              <FeatureCard
                icon={<IconTracking />}
                title="Player Tracking"
                description="Every player tracked. Speed, distance, sprints, fatigue score."
              />
            </FadeInSection>
            <FadeInSection delay={0.2}>
              <FeatureCard
                icon={<IconTeam />}
                title="Team Analysis"
                description="Formation, pressing patterns, defensive shape, tactical vulnerabilities."
              />
            </FadeInSection>
            <FadeInSection delay={0.3}>
              <FeatureCard
                icon={<ShieldCheck size={24} />}
                title="Audit-Grade Accuracy"
                description="The pipeline cross-checks itself. Gemini watches the footage. Claude audits the tracking data against the actual frames. Unreliable metrics get flagged, not reported."
              />
            </FadeInSection>
          </div>
        </div>
      </section>

      {/* ===== YOUR PLAYER PROFILE ===== */}
      <section className="py-24 px-6 relative overflow-hidden">
        <div className="absolute top-1/2 right-0 w-[600px] h-[600px] bg-[#00d4aa]/5 rounded-full blur-[160px] pointer-events-none -translate-y-1/2" />

        <div className="max-w-7xl mx-auto relative">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            {/* Left — text */}
            <FadeInSection>
              <p className="text-[#00d4aa] text-sm font-semibold mb-4 uppercase tracking-[0.3em]">
                Player Profiles
              </p>
              <h2 className="text-3xl md:text-4xl font-bold text-white mb-6 leading-tight">
                Every Player Gets a{" "}
                <span className="text-[#00d4aa]">Card</span>
              </h2>
              <p className="text-gray-400 text-lg leading-relaxed mb-6">
                The pipeline tracks each player individually. Speed, distance covered,
                sprint count, defensive actions, positional heatmap — all extracted
                automatically from the footage.
              </p>
              <p className="text-gray-400 text-lg leading-relaxed mb-8">
                When a coach signs in, their players get named profiles that persist
                across matches. Track development over time, compare performance,
                identify who needs rotation.
              </p>
              <div className="flex gap-6 text-sm">
                <div className="border border-white/10 rounded-lg px-5 py-3 bg-[#0d1117]">
                  <span className="text-[#00d4aa] font-bold text-2xl block">22</span>
                  <span className="text-gray-500">Players Tracked</span>
                </div>
                <div className="border border-white/10 rounded-lg px-5 py-3 bg-[#0d1117]">
                  <span className="text-[#00d4aa] font-bold text-2xl block">30fps</span>
                  <span className="text-gray-500">Frame Resolution</span>
                </div>
                <div className="border border-white/10 rounded-lg px-5 py-3 bg-[#0d1117]">
                  <span className="text-[#00d4aa] font-bold text-2xl block">±0.3m</span>
                  <span className="text-gray-500">Position Accuracy</span>
                </div>
              </div>
            </FadeInSection>

            {/* Right — 3D player model */}
            <FadeInSection delay={0.2}>
              <div className="h-[550px] rounded-2xl border border-white/5 bg-[#0a0f1a] overflow-hidden relative">
                <PlayerModel3D playerName="ATHLINK" playerNumber={10} />
                <div className="absolute bottom-4 left-0 right-0 text-center">
                  <span className="text-gray-600 text-xs tracking-wider">DRAG TO ROTATE &middot; SCROLL TO ZOOM</span>
                </div>
              </div>
            </FadeInSection>
          </div>
        </div>
      </section>

      {/* ===== THE PROBLEM WE SOLVE ===== */}
      <section className="py-24 px-6 relative">
        <div className="max-w-4xl mx-auto">
          <FadeInSection>
            <div className="rounded-2xl bg-[#111827] border border-white/5 p-8 md:p-14">
              <h2 className="text-3xl md:text-4xl font-bold mb-10">
                Grassroots Football Has a{" "}
                <span className="text-[#00d4aa]">Data Problem</span>
              </h2>

              <p className="text-gray-300 text-lg leading-relaxed mb-6">
                Professional clubs spend millions on performance analysis. They know
                exactly how far each player ran, where the team lost possession,
                which pressing triggers broke down, and how individual decision
                quality holds up under pressure. Their coaches walk into half time
                with data. Yours walk in with a feeling.
              </p>

              <p className="text-gray-300 text-lg leading-relaxed mb-6">
                Athlink CV closes that gap. Upload any clip — broadcast footage, a
                phone recording from the touchline, anything. The system watches the
                footage with AI, tracks every visible player, separates teams
                automatically, and builds a complete coaching report. Formation
                analysis. Individual player cards. Pressing pattern breakdown.
                Training recommendations based on what the data actually shows.
              </p>

              <p className="text-gray-300 text-lg leading-relaxed">
                This isn&apos;t a highlight reel tool. It&apos;s not a video library.
                It&apos;s the analytical infrastructure that professional clubs pay
                for, rebuilt for the coaches who actually need it.
              </p>
            </div>
          </FadeInSection>
        </div>
      </section>

      {/* ===== FOOTER ===== */}
      <footer className="border-t border-white/5 py-16 px-6">
        <div className="max-w-5xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-6">
          <div>
            <span className="text-2xl font-bold tracking-wider text-[#00d4aa]">
              ATHLINK CV
            </span>
            <p className="text-gray-500 text-sm mt-2">
              Built for grassroots and semi-professional coaches.
            </p>
          </div>
          <a href={APP_URL} className="cta-button !text-sm">
            START ANALYSING →
          </a>
        </div>
      </footer>
    </main>
  );
}
