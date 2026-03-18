"""
Match Oracle — synthesises a Tactical Fingerprint from multiple
opponent clips that have already been through the analysis pipeline.
"""
import json
import logging
import os
import urllib.request
from collections import Counter

logger = logging.getLogger(__name__)


class MatchOracle:

    def synthesise(self, analyses, opponent_name="Opponent"):
        """
        Takes 2-3 completed analysis result dicts, extracts consistent
        patterns, and returns a Tactical Fingerprint.
        """
        signals = []
        for a in analyses:
            sig = {
                "team_shape": a.get("shape", {}),
                "team_separation": a.get("team_separation", {}),
                "physical": a.get("physical", {}),
                "situations": a.get("situations", {}),
                "brain": a.get("brain", {}),
                "corrections": a.get("corrections_applied", {}),
            }
            signals.append(sig)

        fingerprint_data = self._extract_consistent_patterns(signals)
        report = self._generate_oracle_report(fingerprint_data, opponent_name)

        return {
            "opponent": opponent_name,
            "clips_analysed": len(analyses),
            "fingerprint": fingerprint_data,
            "oracle_report": report,
            "confidence": fingerprint_data.get("overall_confidence"),
        }

    def _extract_consistent_patterns(self, signals):
        """
        Extract what is CONSISTENT across clips.
        Only report a pattern if it appears in 2+ out of N clips.
        Consistency = reliability.
        """
        # Team width — only from clips where brain trusts shape
        widths_0, widths_1 = [], []
        for s in signals:
            shape = s.get("team_shape", {})
            brain = s.get("brain", {})
            trust = brain.get("metrics_to_trust", [])
            if "team_shape" in trust:
                t0 = shape.get("team_0", {})
                t1 = shape.get("team_1", {})
                if t0.get("avg_width_metres"):
                    widths_0.append(t0["avg_width_metres"])
                if t1.get("avg_width_metres"):
                    widths_1.append(t1["avg_width_metres"])

        avg_width_0 = round(sum(widths_0) / len(widths_0), 1) if widths_0 else None
        avg_width_1 = round(sum(widths_1) / len(widths_1), 1) if widths_1 else None

        # Sprint intensity — only if brain does not question it
        sprint_counts = []
        for s in signals:
            phys = s.get("physical", {})
            brain = s.get("brain", {})
            question = brain.get("metrics_to_question", [])
            if "sprint_counts" not in question:
                sc = phys.get("total_sprints")
                if sc is not None:
                    sprint_counts.append(sc)

        sprint_consistent = None
        if len(sprint_counts) >= 2:
            sprint_consistent = round(sum(sprint_counts) / len(sprint_counts), 1)

        # Dominant phase across clips
        phase_patterns = []
        for s in signals:
            sits = s.get("situations", {})
            counts = sits.get("counts", {})
            if counts:
                dominant = max(counts, key=counts.get)
                phase_patterns.append(dominant)

        dominant_phase = None
        if phase_patterns:
            dominant_phase = Counter(phase_patterns).most_common(1)[0][0]

        # Max speed — median across clips
        speeds = []
        for s in signals:
            phys = s.get("physical", {})
            spd = phys.get("max_speed_kmh")
            if spd and spd > 0:
                speeds.append(spd)
        speeds.sort()
        median_speed = speeds[len(speeds) // 2] if speeds else None

        # Overall confidence from brain health scores
        health_scores = []
        for s in signals:
            brain = s.get("brain", {})
            health = brain.get("tracking_health", {})
            reliability = health.get("data_reliability", "low")
            score = {"high": 1.0, "medium": 0.6, "low": 0.3}.get(reliability, 0.3)
            health_scores.append(score)

        overall_conf = round(sum(health_scores) / len(health_scores), 2) if health_scores else 0.3
        conf_label = "high" if overall_conf >= 0.75 else ("medium" if overall_conf >= 0.5 else "low")

        return {
            "avg_team_width_metres": {
                "team_0": avg_width_0,
                "team_1": avg_width_1,
            },
            "sprint_intensity": sprint_consistent,
            "dominant_phase": dominant_phase,
            "median_max_speed_kmh": median_speed,
            "overall_confidence": conf_label,
            "clips_with_trusted_shape": len(widths_0),
            "clips_with_trusted_sprints": len(sprint_counts),
        }

    def _generate_oracle_report(self, fingerprint_data, opponent_name):
        """
        Single Claude call to generate the Tactical Fingerprint report.
        Uses Haiku for fast, cheap synthesis.
        """
        conf = fingerprint_data.get("overall_confidence", "medium")
        width_0 = fingerprint_data.get("avg_team_width_metres", {}).get("team_0")
        width_1 = fingerprint_data.get("avg_team_width_metres", {}).get("team_1")
        sprints = fingerprint_data.get("sprint_intensity")
        phase = fingerprint_data.get("dominant_phase")
        speed = fingerprint_data.get("median_max_speed_kmh")

        prompt = (
            f"You are an elite football analyst writing a pre-match opponent "
            f"intelligence report for a coach.\n\n"
            f"You have analysed {fingerprint_data.get('clips_with_trusted_shape', 0) + fingerprint_data.get('clips_with_trusted_sprints', 0)} "
            f"data points across multiple clips of {opponent_name}.\n"
            f"Data confidence: {conf}\n\n"
            f"CONSISTENT PATTERNS DETECTED:\n"
            f"- Team width (team A): {width_0}m average across clips\n"
            f"- Team width (team B): {width_1}m average across clips\n"
            f"- Sprint intensity: {sprints} sprints per clip average\n"
            f"- Dominant match phase: {phase}\n"
            f"- Peak speed recorded: {speed} km/h\n\n"
            f"Write a Tactical Fingerprint report with these exact sections:\n\n"
            f"## HOW THEY SET UP\n"
            f"2-3 sentences on shape and structure based on width data.\n"
            f"Only make claims supported by the data above.\n"
            f"If confidence is low, hedge appropriately.\n\n"
            f"## HOW THEY PLAY\n"
            f"2-3 sentences on style — press intensity, tempo, transition speed.\n"
            f"Base this on sprint intensity and phase dominance data.\n\n"
            f"## WHERE THEY ARE VULNERABLE\n"
            f"2-3 sentences on exploitable patterns.\n"
            f"Reason from the data — wide team = exposed on transitions,\n"
            f"compact team = may struggle against width.\n"
            f"Be honest if data is insufficient to say.\n\n"
            f"## WHAT TO PREPARE FOR\n"
            f"3 specific tactical instructions the coach should brief their\n"
            f"players on before Saturday.\n"
            f"Make these concrete and actionable.\n\n"
            f"## CONFIDENCE NOTE\n"
            f"One sentence on data reliability and what the coach should\n"
            f"weight more vs less from this report.\n\n"
            f"Do not invent statistics not present in the data above.\n"
            f"Do not use generic football cliches.\n"
            f"Write like a scout who has genuinely watched these clips."
        )

        data = json.dumps({
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=data,
            headers={
                "Content-Type": "application/json",
                "x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""),
                "anthropic-version": "2023-06-01",
            },
        )

        try:
            with urllib.request.urlopen(req) as resp:
                result = json.loads(resp.read())
                return result["content"][0]["text"]
        except Exception as e:
            logger.exception("Oracle report generation failed")
            return f"Oracle report unavailable: {e}"
