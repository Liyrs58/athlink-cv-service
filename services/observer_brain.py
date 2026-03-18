"""
Observer Brain — continuous belief-updating system for match analysis.

Wraps existing services: reads tracking output frame by frame, maintains
a live internal belief state, and provides confidence-aware summaries
for the Claude coaching report.
"""
import logging
import math

logger = logging.getLogger(__name__)


class ObserverBrain:

    def __init__(self):
        self.belief_state = {
            "phase": "unknown",
            "phase_confidence": 0.0,
            "possession_team": None,
            "possession_confidence": 0.0,
            "press_intensity": 0.0,
            "transition_risk": 0.0,
            "tracking_health": 1.0,
            "camera_mode": "unknown",
            "shape": {
                "team_0": {"width": None, "depth": None, "confidence": 0.0},
                "team_1": {"width": None, "depth": None, "confidence": 0.0},
            },
            "anomalies": [],
            "events": [],
            "frame_history": [],
        }
        self.frame_count = 0
        self.last_valid_shape_frame = 0
        self.track_health_history = []
        self._prev_track_count = None
        self._phase_streak = 0
        self._pending_phase = None

    # ------------------------------------------------------------------
    # Main update — called once per frame
    # ------------------------------------------------------------------

    def update(self, frame_idx, active_tracks, frame_metadata, calibration):
        """Process one frame and update belief state in place."""
        self.frame_count += 1
        n_tracks = len(active_tracks)

        # a) tracking health
        health = self._compute_tracking_health(active_tracks)
        self.belief_state["tracking_health"] = health
        self.track_health_history.append({"frame": frame_idx, "health": health, "tracks": n_tracks})

        # b) camera mode
        cam = self._detect_camera_mode(active_tracks, frame_idx)
        self.belief_state["camera_mode"] = cam

        if cam == "cut":
            self.belief_state["tracking_health"] = 0.1
            self.belief_state["anomalies"].append({
                "type": "camera_interruption",
                "frame": frame_idx,
                "detail": f"camera_mode={cam}, {n_tracks} tracks visible",
            })
        elif cam == "close_up":
            # don't update shape but don't treat as hard anomaly —
            # sparse trajectory data often produces low per-frame counts
            self.belief_state["tracking_health"] = min(health, 0.4)
        else:
            # only update shape when camera is usable
            self._update_shape_belief(active_tracks, frame_idx, calibration)

        # c) phase belief
        self._update_phase_belief(active_tracks, frame_idx)

        # d) anomalies
        self._detect_anomalies(active_tracks, frame_idx, calibration)

        # record frame snapshot
        self.belief_state["frame_history"].append({
            "frame": frame_idx,
            "tracks": n_tracks,
            "health": round(health, 2),
            "phase": self.belief_state["phase"],
            "camera": cam,
        })

        self._prev_track_count = n_tracks

    # ------------------------------------------------------------------
    # Sub-routines
    # ------------------------------------------------------------------

    def _compute_tracking_health(self, active_tracks):
        """Return 0-1 health score based on track count and stability."""
        n = len(active_tracks)

        # base score from count — tuned for sparse trajectory data
        # (not every track has an entry at every frame due to frame_stride)
        if n >= 10:
            base = 1.0
        elif n >= 5:
            base = 0.6 + (n - 5) / 12.5  # 0.6 → 1.0
        elif n >= 1:
            base = 0.3 + (n - 1) / 13.3  # 0.3 → 0.6
        else:
            base = 0.0

        # penalise instability (big jumps in track count)
        if self._prev_track_count is not None:
            delta = abs(n - self._prev_track_count)
            if delta > 8:
                base *= 0.4
            elif delta > 4:
                base *= 0.7

        return round(min(max(base, 0.0), 1.0), 3)

    def _detect_camera_mode(self, active_tracks, frame_idx):
        n = len(active_tracks)
        if n == 0:
            return "cut"
        if n < 3:
            return "close_up"
        if n < 8:
            return "medium_broadcast"
        return "wide_broadcast"

    def _update_phase_belief(self, active_tracks, frame_idx):
        """Update phase with 3-frame confirmation to avoid flicker."""
        n = len(active_tracks)
        health = self.belief_state["tracking_health"]

        # determine candidate phase
        if n < 2 and health < 0.3:
            candidate = "dead_ball"
        elif n >= 8:
            candidate = "open_play"
        else:
            candidate = self.belief_state["phase"]  # hold current

        # streak logic — require 3 consecutive frames to flip
        if candidate == self._pending_phase:
            self._phase_streak += 1
        else:
            self._pending_phase = candidate
            self._phase_streak = 1

        if self._phase_streak >= 3 and candidate != self.belief_state["phase"]:
            # record event
            self.belief_state["events"].append({
                "type": "phase_change",
                "frame": frame_idx,
                "from": self.belief_state["phase"],
                "to": candidate,
            })
            self.belief_state["phase"] = candidate
            self.belief_state["phase_confidence"] = 0.6

        # nudge confidence
        if candidate == self.belief_state["phase"]:
            self.belief_state["phase_confidence"] = min(
                1.0, self.belief_state["phase_confidence"] + 0.05
            )
        else:
            self.belief_state["phase_confidence"] = max(
                0.3, self.belief_state["phase_confidence"] - 0.03
            )

    def _update_shape_belief(self, active_tracks, frame_idx, calibration):
        """Update per-team shape width/depth from track positions."""
        ppm = 1.0
        if calibration and calibration.get("pixels_per_metre"):
            ppm = max(calibration["pixels_per_metre"], 0.1)

        team_tracks = {"team_0": [], "team_1": []}
        for t in active_tracks:
            tid = t.get("teamId", -1)
            bbox = t.get("bbox")
            if bbox and tid in (0, 1):
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                key = f"team_{tid}"
                team_tracks[key].append((cx / ppm, cy / ppm))

        for key in ("team_0", "team_1"):
            pts = team_tracks[key]
            if len(pts) >= 3:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                width = max(xs) - min(xs)
                depth = max(ys) - min(ys)
                self.belief_state["shape"][key]["width"] = round(width, 1)
                self.belief_state["shape"][key]["depth"] = round(depth, 1)
                self.belief_state["shape"][key]["confidence"] = min(
                    1.0, self.belief_state["shape"][key]["confidence"] + 0.05
                )
                self.last_valid_shape_frame = frame_idx

    def _detect_anomalies(self, active_tracks, frame_idx, calibration):
        """Check for anomalies and append to belief_state."""
        n = len(active_tracks)
        anomalies = self.belief_state["anomalies"]

        # track count jump
        if self._prev_track_count is not None:
            delta = abs(n - self._prev_track_count)
            if delta > 8:
                anomalies.append({
                    "type": "track_instability",
                    "frame": frame_idx,
                    "detail": f"count jumped by {delta} ({self._prev_track_count} → {n})",
                })

        # overcounting
        if n > 25:
            anomalies.append({
                "type": "overcounting",
                "frame": frame_idx,
                "detail": f"{n} active tracks",
            })

        # speed spikes (check bbox movement if previous frame data available)
        ppm = 1.0
        if calibration and calibration.get("pixels_per_metre"):
            ppm = max(calibration["pixels_per_metre"], 0.1)

        for t in active_tracks:
            speed = t.get("speed_ms")
            if speed is not None and speed > 10.0:
                anomalies.append({
                    "type": "speed_spike",
                    "frame": frame_idx,
                    "detail": f"track_{t.get('trackId', '?')} at {speed:.1f} m/s",
                })

        # shape geometry check
        for key in ("team_0", "team_1"):
            w = self.belief_state["shape"][key].get("width")
            if w is not None and w > 68:
                anomalies.append({
                    "type": "geometry_unreliable",
                    "frame": frame_idx,
                    "detail": f"{key} width {w}m exceeds pitch width",
                })

    # ------------------------------------------------------------------
    # Summary methods
    # ------------------------------------------------------------------

    def get_tracking_health_summary(self):
        """Overall tracking health across the entire clip."""
        if not self.track_health_history:
            return {
                "average_health": 0.0,
                "worst_phase": "no data",
                "best_phase": "no data",
                "camera_cuts_detected": 0,
                "total_anomalies": len(self.belief_state["anomalies"]),
                "data_reliability": "low",
            }

        healths = [h["health"] for h in self.track_health_history]
        avg = sum(healths) / len(healths)

        # find worst and best contiguous runs
        worst_start, worst_end, worst_avg = self._find_extreme_run(healths, worst=True)
        best_start, best_end, best_avg = self._find_extreme_run(healths, worst=False)

        frames = [h["frame"] for h in self.track_health_history]
        worst_frames = f"frames {frames[worst_start]}-{frames[min(worst_end, len(frames) - 1)]}"
        best_frames = f"frames {frames[best_start]}-{frames[min(best_end, len(frames) - 1)]}"

        camera_cuts = sum(
            1 for a in self.belief_state["anomalies"]
            if a["type"] == "camera_interruption"
        )

        if avg >= 0.7:
            reliability = "high"
        elif avg >= 0.4:
            reliability = "medium"
        else:
            reliability = "low"

        return {
            "average_health": round(avg, 2),
            "worst_phase": worst_frames,
            "best_phase": best_frames,
            "camera_cuts_detected": camera_cuts,
            "total_anomalies": len(self.belief_state["anomalies"]),
            "data_reliability": reliability,
        }

    def _find_extreme_run(self, values, worst=True, window=10):
        """Find the contiguous window with the worst (or best) average."""
        if len(values) <= window:
            avg = sum(values) / len(values)
            return 0, len(values) - 1, avg

        best_avg = None
        best_start = 0
        for i in range(len(values) - window + 1):
            chunk = values[i:i + window]
            a = sum(chunk) / len(chunk)
            if best_avg is None:
                best_avg = a
                best_start = i
            elif worst and a < best_avg:
                best_avg = a
                best_start = i
            elif not worst and a > best_avg:
                best_avg = a
                best_start = i

        return best_start, best_start + window - 1, best_avg

    def get_belief_summary(self):
        """Final summary for passing to the Claude coaching prompt."""
        health_summary = self.get_tracking_health_summary()

        # build match phases from events
        match_phases = self._build_match_phases()

        # anomaly counts by type
        anomaly_counts = {}
        for a in self.belief_state["anomalies"]:
            anomaly_counts[a["type"]] = anomaly_counts.get(a["type"], 0) + 1

        anomaly_parts = []
        for atype in ("camera_interruption", "speed_spike", "overcounting", "track_instability", "geometry_unreliable"):
            count = anomaly_counts.get(atype, 0)
            label = atype.replace("_", " ")
            anomaly_parts.append(f"{count} {label}{'s' if count != 1 else ''}")
        anomalies_summary = ", ".join(anomaly_parts)

        # decide what to trust / question
        reliability = health_summary["data_reliability"]
        camera_cuts = health_summary["camera_cuts_detected"]

        metrics_to_trust = []
        metrics_to_question = []

        if reliability in ("high", "medium"):
            metrics_to_trust.extend(["situation_timeline", "team_shape", "player_count"])
        else:
            metrics_to_question.extend(["situation_timeline", "team_shape"])
            metrics_to_trust.append("player_count")

        speed_spikes = anomaly_counts.get("speed_spike", 0)
        if speed_spikes > 3:
            metrics_to_question.append("individual_speeds")
        else:
            metrics_to_trust.append("individual_speeds")

        if camera_cuts > 2:
            metrics_to_question.extend(["individual_distances", "sprint_counts"])
        else:
            if reliability == "high":
                metrics_to_trust.extend(["individual_distances", "sprint_counts"])
            else:
                metrics_to_question.extend(["individual_distances", "sprint_counts"])

        # build verdict
        verdict = self._build_verdict(health_summary, anomaly_counts)

        return {
            "match_phases": match_phases,
            "tracking_health": health_summary,
            "anomalies_summary": anomalies_summary,
            "metrics_to_trust": metrics_to_trust,
            "metrics_to_question": metrics_to_question,
            "brain_verdict": verdict,
        }

    def _build_match_phases(self):
        """Convert frame_history into contiguous phase segments."""
        history = self.belief_state["frame_history"]
        if not history:
            return []

        phases = []
        current_phase = history[0]["phase"]
        start_frame = history[0]["frame"]
        confidences = [self.belief_state["phase_confidence"]]

        for entry in history[1:]:
            if entry["phase"] != current_phase:
                # close previous
                end_time = start_frame / 25.0  # approximate
                phases.append({
                    "phase": current_phase,
                    "start": round(start_frame / 25.0, 1),
                    "end": round(entry["frame"] / 25.0, 1),
                    "confidence": round(sum(confidences) / len(confidences), 2),
                })
                current_phase = entry["phase"]
                start_frame = entry["frame"]
                confidences = []
            confidences.append(entry.get("health", 0.5))

        # close last
        if history:
            phases.append({
                "phase": current_phase,
                "start": round(start_frame / 25.0, 1),
                "end": round(history[-1]["frame"] / 25.0, 1),
                "confidence": round(sum(confidences) / max(len(confidences), 1), 2),
            })

        return phases

    def _build_verdict(self, health_summary, anomaly_counts):
        """Generate a human-readable verdict string."""
        reliability = health_summary["data_reliability"]
        avg = health_summary["average_health"]
        cuts = health_summary["camera_cuts_detected"]
        total_anomalies = health_summary["total_anomalies"]

        parts = []

        if reliability == "high":
            parts.append("Good quality clip.")
        elif reliability == "medium":
            parts.append("Acceptable quality clip with some tracking gaps.")
        else:
            parts.append("Low quality tracking — interpret all metrics with caution.")

        if cuts > 0:
            parts.append(f"{cuts} camera cut{'s' if cuts != 1 else ''} detected.")

        if total_anomalies > 5:
            parts.append(f"{total_anomalies} anomalies logged — some metrics may be affected.")

        if reliability in ("high", "medium"):
            parts.append("Team-level metrics reliable.")
        else:
            parts.append("Team-level metrics approximate.")

        speed_spikes = anomaly_counts.get("speed_spike", 0)
        if speed_spikes > 3:
            parts.append("Individual speed data has noise — use ranges.")
        elif reliability == "high":
            parts.append("Individual stats have high confidence.")
        else:
            parts.append("Individual stats have medium confidence.")

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Entry point — process entire match
    # ------------------------------------------------------------------

    def process_full_match(self, tracks, frame_metadata, calibration):
        """
        Iterate through all frames, build per-frame active track lists,
        call update() for each, then return the final belief summary.
        """
        # Build a lookup: frame_idx → list of active tracks with bbox
        # A track is active at frame_idx if firstSeen <= frame_idx <= lastSeen
        # and it has a trajectory entry near that frame.
        sorted_meta = sorted(frame_metadata, key=lambda m: m.get("frameIndex", 0))

        for meta in sorted_meta:
            frame_idx = meta.get("frameIndex", 0)

            active = []
            for t in tracks:
                first = t.get("firstSeen", 0)
                last = t.get("lastSeen", 0)
                if first <= frame_idx <= last:
                    # find closest trajectory entry
                    traj = t.get("trajectory", [])
                    closest = None
                    if traj:
                        closest = min(traj, key=lambda e: abs(e["frameIndex"] - frame_idx))
                    if closest and abs(closest["frameIndex"] - frame_idx) <= 4:
                        active.append({
                            "trackId": t.get("trackId"),
                            "teamId": t.get("teamId", -1),
                            "bbox": closest.get("bbox"),
                            "speed_ms": closest.get("speed_ms"),
                        })

            self.update(frame_idx, active, meta, calibration)

        return self.get_belief_summary()
