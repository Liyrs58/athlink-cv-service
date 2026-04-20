"""Player fatigue clock tracking based on distance and sprints.
"""

import math
import cmath
from typing import List, Dict, Any, Optional
from statistics import median

SPRINT_MS = 5.5        # 19.8 km/h — sprint threshold
PIXELS_PER_METRE = 15.5
MAX_REALISTIC_SPEED_MS = 10.0


class FatigueClock:

    def _compute_speeds(self, traj: list) -> list:
        """
        Extracts smoothed speed signal (m/s) from a trajectory.
        Uses world_x/world_y if available, falls back to bbox centroids.
        """
        if len(traj) < 2:
            return []

        speeds = []
        for i in range(1, len(traj)):
            prev = traj[i - 1]
            curr = traj[i]
            dt = curr.get("timestampSeconds", 0) - prev.get("timestampSeconds", 0)
            if dt <= 0:
                speeds.append(0.0)
                continue

            if "world_x" in curr and "world_x" in prev:
                dx = curr["world_x"] - prev["world_x"]
                dy = curr["world_y"] - prev["world_y"]
                dist_m = math.sqrt(dx * dx + dy * dy)
            else:
                bbox_c = curr.get("bbox", [0, 0, 0, 0])
                bbox_p = prev.get("bbox", [0, 0, 0, 0])
                cx_c = (bbox_c[0] + bbox_c[2]) / 2.0
                cy_c = (bbox_c[1] + bbox_c[3]) / 2.0
                cx_p = (bbox_p[0] + bbox_p[2]) / 2.0
                cy_p = (bbox_p[1] + bbox_p[3]) / 2.0
                dist_px = math.sqrt((cx_c - cx_p) ** 2 + (cy_c - cy_p) ** 2)
                dist_m = dist_px / PIXELS_PER_METRE

            speed = dist_m / dt
            speeds.append(min(speed, MAX_REALISTIC_SPEED_MS))

        # Rolling median smoothing (window=3) to reduce jitter
        if len(speeds) >= 3:
            smoothed = []
            for i in range(len(speeds)):
                lo = max(0, i - 1)
                hi = min(len(speeds), i + 2)
                smoothed.append(median(speeds[lo:hi]))
            return smoothed

        return speeds

    def _compute_dft(self, signal: list) -> list:
        """
        Pure Python Discrete Fourier Transform.
        No numpy dependency — works on Railway CPU.
        Returns list of (frequency, amplitude) pairs
        sorted by amplitude descending.
        """
        n = len(signal)
        if n < 4:
            return []

        result = []
        for k in range(n // 2):  # only need first half (Nyquist)
            total = complex(0, 0)
            for t, x in enumerate(signal):
                angle = -2 * cmath.pi * k * t / n
                total += x * cmath.exp(complex(0, angle))
            amplitude = abs(total) / n
            frequency = k  # frequency bin index
            result.append((frequency, round(amplitude, 4)))

        return sorted(result, key=lambda x: x[1], reverse=True)

    def _fourier_fatigue_metrics(self, speeds: list) -> dict:
        """
        Analyses the frequency spectrum of a player's speed signal.

        Key insight:
        - Fresh player: speed signal has HIGH frequency components
          (sharp accelerations, varied rhythm, unpredictable)
        - Fatigued player: speed signal has LOW frequency components
          (slow oscillations, predictable rhythm, grinding)

        Returns fatigue indicators from the frequency domain.
        """
        if len(speeds) < 8:
            return {
                "dominant_frequency": None,
                "high_freq_ratio": 0.0,
                "spectral_entropy": 0.0,
                "fatigue_frequency_score": 0,
            }

        spectrum = self._compute_dft(speeds)
        if not spectrum:
            return {
                "dominant_frequency": None,
                "high_freq_ratio": 0.0,
                "spectral_entropy": 0.0,
                "fatigue_frequency_score": 0,
            }

        # Total power
        total_power = sum(amp for _, amp in spectrum)
        if total_power == 0:
            return {
                "dominant_frequency": None,
                "high_freq_ratio": 0.0,
                "spectral_entropy": 0.0,
                "fatigue_frequency_score": 0,
            }

        # Dominant frequency (highest amplitude)
        dominant_freq = spectrum[0][0]

        # High frequency ratio
        # Frequencies above n/4 are "high frequency"
        n = len(speeds)
        high_freq_threshold = n // 4
        high_freq_power = sum(
            amp for freq, amp in spectrum if freq >= high_freq_threshold
        )
        high_freq_ratio = round(high_freq_power / total_power, 3)

        # Spectral entropy — how spread is the power?
        # Low spectral entropy = power concentrated in few frequencies
        # (monotone, predictable movement = fatigue signature)
        spectral_entropy = 0.0
        for _, amp in spectrum:
            p = amp / total_power
            if p > 0:
                spectral_entropy -= p * cmath.log(complex(p)).real
        max_spec_entropy = cmath.log(complex(len(spectrum))).real
        spectral_entropy = round(
            spectral_entropy / max_spec_entropy if max_spec_entropy > 0 else 0, 3
        )

        # Fatigue frequency score 0-100
        # Low high_freq_ratio = fatigued (dominated by slow oscillations)
        # Low spectral_entropy = fatigued (monotone movement)
        freq_fatigue = (1 - high_freq_ratio) * 50
        entropy_fatigue = (1 - spectral_entropy) * 50
        fatigue_frequency_score = int(round(freq_fatigue + entropy_fatigue))

        return {
            "dominant_frequency": dominant_freq,
            "high_freq_ratio": high_freq_ratio,
            "spectral_entropy": spectral_entropy,
            "fatigue_frequency_score": fatigue_frequency_score,
        }

    def _compute_fatigue_score(self, vel_drop_pct, sprint_drift_pct,
                                recovery_expansion, predictability,
                                fourier_score=0) -> int:
        """
        Updated weights including Fourier score:
        Velocity ceiling drop:    30% weight
        Sprint threshold drift:   25% weight
        Recovery expansion:       15% weight
        Predictability:           10% weight
        Fourier frequency score:  20% weight  <- new
        """
        vel_score = min(100, max(0, vel_drop_pct * 2))
        sprint_score = min(100, max(0, sprint_drift_pct * 1.5))
        recovery_score = min(100, max(0, recovery_expansion * 15))
        pred_score = min(100, max(0, predictability))
        freq_score = min(100, max(0, fourier_score))

        score = (
            vel_score * 0.30 +
            sprint_score * 0.25 +
            recovery_score * 0.15 +
            pred_score * 0.10 +
            freq_score * 0.20
        )
        return int(round(score))

    def analyse_player(self, track: Dict[str, Any], calibration: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """
        Analyses a single player track for fatigue indicators.
        Returns None if insufficient data.
        """
        traj = track.get("trajectory", [])
        if len(traj) < 10:
            return None

        track_id = track.get("trackId")
        team_id = track.get("teamId", -1)

        # Split into first and second half
        mid = len(traj) // 2
        first_half = traj[:mid]
        second_half = traj[mid:]

        first_speeds = self._compute_speeds(first_half)
        second_speeds = self._compute_speeds(second_half)

        if not first_speeds or not second_speeds:
            return None

        # Velocity ceiling drop — max speed first half vs second half
        first_max = max(first_speeds) if first_speeds else 0.0
        second_max = max(second_speeds) if second_speeds else 0.0
        if first_max > 0:
            vel_drop_pct = max(0.0, (first_max - second_max) / first_max * 100)
        else:
            vel_drop_pct = 0.0

        # Sprint threshold drift — how much sprint threshold drops
        first_sprint_speeds = [s for s in first_speeds if s >= SPRINT_MS]
        second_sprint_speeds = [s for s in second_speeds if s >= SPRINT_MS]
        first_sprint_avg = sum(first_sprint_speeds) / len(first_sprint_speeds) if first_sprint_speeds else 0.0
        second_sprint_avg = sum(second_sprint_speeds) / len(second_sprint_speeds) if second_sprint_speeds else 0.0
        if first_sprint_avg > 0:
            sprint_drift_pct = max(0.0, (first_sprint_avg - second_sprint_avg) / first_sprint_avg * 100)
        else:
            sprint_drift_pct = 0.0

        # Recovery expansion — time between sprint bursts increasing
        all_speeds = self._compute_speeds(traj)
        recovery_times = []
        in_sprint = False
        last_sprint_end_idx = None
        for i, s in enumerate(all_speeds):
            if s >= SPRINT_MS:
                if not in_sprint:
                    if last_sprint_end_idx is not None:
                        gap = i - last_sprint_end_idx
                        recovery_times.append(gap)
                    in_sprint = True
            else:
                if in_sprint:
                    last_sprint_end_idx = i
                    in_sprint = False

        recovery_expansion = 0.0
        if len(recovery_times) >= 4:
            first_recoveries = recovery_times[:len(recovery_times) // 2]
            last_recoveries = recovery_times[len(recovery_times) // 2:]
            first_avg_rec = sum(first_recoveries) / len(first_recoveries)
            last_avg_rec = sum(last_recoveries) / len(last_recoveries)
            # Convert frame gaps to seconds (frame_stride=2, fps=25 → 0.08s per frame)
            recovery_expansion = max(0.0, (last_avg_rec - first_avg_rec) * 0.08)

        # Predictability — coefficient of variation of speed (inverted)
        # Low CV = predictable, monotone = fatigue signal
        if all_speeds:
            avg_s = sum(all_speeds) / len(all_speeds)
            if avg_s > 0:
                variance = sum((s - avg_s) ** 2 for s in all_speeds) / len(all_speeds)
                std_s = math.sqrt(variance)
                cv = std_s / avg_s  # coefficient of variation
                # High CV = unpredictable (fresh). Low CV = predictable (fatigued).
                predictability = max(0.0, (1 - min(cv, 1.0)) * 100)
            else:
                predictability = 50.0
        else:
            predictability = 50.0

        # Fourier analysis on full speed signal
        fourier_metrics = self._fourier_fatigue_metrics(all_speeds)

        fatigue_score = self._compute_fatigue_score(
            vel_drop_pct,
            sprint_drift_pct,
            recovery_expansion,
            predictability,
            fourier_score=fourier_metrics.get("fatigue_frequency_score", 0),
        )

        if fatigue_score >= 70:
            fatigue_label = "HIGH"
        elif fatigue_score >= 40:
            fatigue_label = "MEDIUM"
        else:
            fatigue_label = "LOW"

        return {
            "track_id": track_id,
            "team_id": team_id,
            "fatigue_score": fatigue_score,
            "fatigue_label": fatigue_label,
            "metrics": {
                "velocity_ceiling_drop_pct": round(vel_drop_pct, 1),
                "sprint_threshold_drift_pct": round(sprint_drift_pct, 1),
                "recovery_expansion_seconds": round(recovery_expansion, 2),
                "predictability_score": round(predictability, 1),
            },
            "fourier": fourier_metrics,
        }

    def analyse_all_players(
        self,
        tracks: List[Dict[str, Any]],
        calibration: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Runs fatigue analysis across all confirmed player tracks.
        Returns profiles, team averages, and most fatigued player.
        """
        profiles = []
        for t in tracks:
            if t.get("is_staff", False):
                continue
            if t.get("confirmed_detections", 0) < 5:
                continue
            result = self.analyse_player(t, calibration=calibration)
            if result is not None:
                profiles.append(result)

        if not profiles:
            return {
                "status": "insufficient_data",
                "players_analysed": 0,
                "profiles": [],
            }

        # Team averages
        team_scores: Dict[int, List[float]] = {}
        for p in profiles:
            tid = p["team_id"]
            team_scores.setdefault(tid, []).append(p["fatigue_score"])

        team_fatigue_averages = {
            str(tid): round(sum(scores) / len(scores), 1)
            for tid, scores in team_scores.items()
        }

        most_fatigued = max(profiles, key=lambda p: p["fatigue_score"])

        # Clock confidence based on trajectory lengths
        avg_traj_len = sum(
            len(t.get("trajectory", []))
            for t in tracks
            if not t.get("is_staff", False) and t.get("confirmed_detections", 0) >= 5
        ) / max(len(profiles), 1)

        if avg_traj_len >= 50:
            clock_confidence = "high"
        elif avg_traj_len >= 20:
            clock_confidence = "medium"
        else:
            clock_confidence = "low"

        return {
            "status": "ok",
            "players_analysed": len(profiles),
            "clock_confidence": clock_confidence,
            "most_fatigued_track_id": most_fatigued["track_id"],
            "most_fatigued_score": most_fatigued["fatigue_score"],
            "team_fatigue_averages": team_fatigue_averages,
            "profiles": profiles,
        }
