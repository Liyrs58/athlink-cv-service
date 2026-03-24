"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { motion, useInView } from "framer-motion";
import { GlowingEffect } from "@/components/ui/glowing-effect";

const APP_URL = "https://web-production-6b7ca.up.railway.app";

/* ============================================================
   ANIMATED FOOTBALL PITCH
   ============================================================ */
// Team A (teal) — 4-2 formation: 4 defenders, 2 midfielders
const TEAM_A_POSITIONS = [
  // Phase 1
  [
    { x: 20, y: 65 }, { x: 40, y: 62 }, { x: 60, y: 62 }, { x: 80, y: 65 },
    { x: 35, y: 50 }, { x: 65, y: 50 },
  ],
  // Phase 2 — shift right
  [
    { x: 22, y: 63 }, { x: 42, y: 58 }, { x: 62, y: 60 }, { x: 82, y: 63 },
    { x: 40, y: 46 }, { x: 60, y: 44 },
  ],
  // Phase 3 — press higher
  [
    { x: 18, y: 60 }, { x: 38, y: 55 }, { x: 58, y: 56 }, { x: 78, y: 60 },
    { x: 38, y: 42 }, { x: 62, y: 40 },
  ],
  // Phase 4 — drop back
  [
    { x: 20, y: 68 }, { x: 40, y: 66 }, { x: 60, y: 66 }, { x: 80, y: 68 },
    { x: 36, y: 54 }, { x: 64, y: 54 },
  ],
];

// Team B (white) — 4-3 formation
const TEAM_B_POSITIONS = [
  [
    { x: 20, y: 35 }, { x: 40, y: 38 }, { x: 60, y: 38 }, { x: 80, y: 35 },
    { x: 30, y: 48 }, { x: 50, y: 45 }, 
  ],
  [
    { x: 22, y: 37 }, { x: 38, y: 42 }, { x: 58, y: 40 }, { x: 78, y: 37 },
    { x: 32, y: 52 }, { x: 55, y: 48 },
  ],
  [
    { x: 18, y: 40 }, { x: 42, y: 44 }, { x: 62, y: 42 }, { x: 82, y: 40 },
    { x: 28, y: 54 }, { x: 52, y: 52 },
  ],
  [
    { x: 20, y: 33 }, { x: 40, y: 36 }, { x: 60, y: 36 }, { x: 80, y: 33 },
    { x: 34, y: 46 }, { x: 48, y: 42 },
  ],
];

const BALL_POSITIONS = [
  { x: 50, y: 50 },
  { x: 42, y: 44 },
  { x: 60, y: 40 },
  { x: 55, y: 52 },
  { x: 38, y: 55 },
  { x: 62, y: 48 },
  { x: 48, y: 42 },
  { x: 52, y: 54 },
];

function AnimatedPitch() {
  const [phase, setPhase] = useState(0);
  const [ballIdx, setBallIdx] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setPhase((p) => (p + 1) % TEAM_A_POSITIONS.length);
      setBallIdx((b) => (b + 1) % BALL_POSITIONS.length);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  const teamA = TEAM_A_POSITIONS[phase];
  const teamB = TEAM_B_POSITIONS[phase];
  const ball = BALL_POSITIONS[ballIdx];

  return (
    <div className="pitch-container mx-auto">
      <div className="pitch">
        {/* Pitch markings */}
        <div className="pitch-markings">
          <div className="halfway-line" />
          <div className="centre-circle" />
          <div className="centre-spot" />
          <div className="penalty-box-top" />
          <div className="goal-box-top" />
          <div className="penalty-box-bottom" />
          <div className="goal-box-bottom" />
          <div className="corner-tl" />
          <div className="corner-tr" />
          <div className="corner-bl" />
          <div className="corner-br" />
        </div>

        {/* Team A dots (teal) */}
        {teamA.map((pos, i) => (
          <div
            key={`a-${i}`}
            className="player-dot team-a"
            style={{ left: `${pos.x}%`, top: `${pos.y}%` }}
          />
        ))}

        {/* Team B dots (white) */}
        {teamB.map((pos, i) => (
          <div
            key={`b-${i}`}
            className="player-dot team-b"
            style={{ left: `${pos.x}%`, top: `${pos.y}%` }}
          />
        ))}

        {/* Ball */}
        <div
          className="player-dot ball"
          style={{ left: `${ball.x}%`, top: `${ball.y}%` }}
        />
      </div>
    </div>
  );
}

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

const IconSpeed = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
  </svg>
);

/* ============================================================
   MAIN PAGE
   ============================================================ */
export default function Home() {
  return (
    <main className="min-h-screen">
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
              TRACAB charges{" "}
              <span className="text-white font-semibold">£10,000+</span> per
              match. Athlink CV is{" "}
              <span className="text-[#00d4aa] font-semibold">£30/month</span>.
              Upload a clip. Get a full coaching report in 90 seconds.
            </p>
            <a href={APP_URL} className="cta-button">
              START ANALYSING →
            </a>
          </FadeInSection>

          {/* Right — animated pitch */}
          <FadeInSection delay={0.3} className="flex justify-center">
            <AnimatedPitch />
          </FadeInSection>
        </div>
      </section>

      {/* ===== HOW IT WORKS ===== */}
      <section className="py-24 px-6 relative">
        <div className="max-w-5xl mx-auto">
          <FadeInSection>
            <h2 className="section-title">How It Works</h2>
            <p className="text-gray-400 text-center mb-16 max-w-xl mx-auto">
              Three steps. Ninety seconds. One complete coaching report.
            </p>
          </FadeInSection>

          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                step: "01",
                title: "Upload Your Clip",
                desc: "Upload any football clip — broadcast or amateur. Any angle, any quality.",
              },
              {
                step: "02",
                title: "AI Pipeline Runs",
                desc: "AI pipeline tracks every player, separates teams, detects the ball frame by frame.",
              },
              {
                step: "03",
                title: "Get Your Report",
                desc: "Full coaching report in 90 seconds — tactics, player cards, training plan.",
              },
            ].map((item, i) => (
              <FadeInSection key={item.step} delay={i * 0.15}>
                <div className="relative p-6 rounded-xl bg-[#111827]/60 border border-white/5 text-center group hover:border-[#00d4aa]/20 transition-all duration-300">
                  <span className="text-5xl font-bold text-[#00d4aa]/15 absolute top-3 left-1/2 -translate-x-1/2">
                    {item.step}
                  </span>
                  <div className="pt-10">
                    <h3 className="text-xl font-bold text-white mb-3">
                      {item.title}
                    </h3>
                    <p className="text-gray-400 text-sm leading-relaxed">
                      {item.desc}
                    </p>
                  </div>
                </div>
              </FadeInSection>
            ))}
          </div>
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
                icon={<IconSpeed />}
                title="90 Second Turnaround"
                description="GPU-accelerated pipeline. Upload at half time, read the report before the second half."
              />
            </FadeInSection>
          </div>
        </div>
      </section>

      {/* ===== COMPETITOR TABLE ===== */}
      <section className="py-24 px-6 relative">
        <div className="max-w-4xl mx-auto">
          <FadeInSection>
            <h2 className="section-title">The Competition</h2>
            <p className="text-gray-400 text-center mb-16 max-w-xl mx-auto">
              See how Athlink CV stacks up against industry players.
            </p>
          </FadeInSection>

          <FadeInSection delay={0.15}>
            <div className="overflow-x-auto rounded-xl bg-[#111827]/40 border border-white/5 p-4">
              <table className="competitor-table">
                <thead>
                  <tr>
                    <th>Platform</th>
                    <th>Price</th>
                    <th>AI Report</th>
                    <th>Auto Tracking</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td className="text-white font-semibold">TRACAB</td>
                    <td className="text-gray-300">£10,000+ / match</td>
                    <td className="text-[#00d4aa]">✓</td>
                    <td className="text-[#00d4aa]">✓</td>
                  </tr>
                  <tr>
                    <td className="text-white font-semibold">Hudl</td>
                    <td className="text-gray-300">£200+ / month</td>
                    <td className="text-red-400">✗</td>
                    <td className="text-red-400">✗</td>
                  </tr>
                  <tr>
                    <td className="text-white font-semibold">Veo</td>
                    <td className="text-gray-300">£99+ / month</td>
                    <td className="text-red-400">✗</td>
                    <td className="text-red-400">✗</td>
                  </tr>
                  <tr className="athlink-row">
                    <td>ATHLINK CV</td>
                    <td>£30 / month</td>
                    <td>✓</td>
                    <td>✓</td>
                  </tr>
                </tbody>
              </table>
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
