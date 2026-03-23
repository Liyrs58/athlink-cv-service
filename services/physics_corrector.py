"""
Physics corrector service.

Applies hard physical and football constraints to raw tracking output
to correct impossible values and maximise accuracy.
"""

import math
import time
import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

def to_scalar(v):
    """Convert numpy scalars to Python native types for JSON serialization."""
    if hasattr(v, 'item'):
        return v.item()
    if hasattr(v, '__float__'):
        return float(v)
    return v

# ── Physical constants ────────────────────────────────────────────
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
PITCH_MARGIN = 5.0  # metres beyond pitch boundary still valid

MAX_SPEED_MS = 11.5        # Usain Bolt peak
MAX_ACCEL_MS2 = 5.0        # world-class athlete
MAX_DECEL_MS2 = 7.0
MAX_STEP_M = 2.8           # max displacement per frame at 25fps
MAX_ACTIVE_TRACKS = 25     # 22 players + ref + margin
COLLISION_DIST_M = 1.5     # two players can't occupy same spot
GAP_FILL_MAX_S = 1.5       # max gap to interpolate
FORMATION_OUTLIER_M = 40.0 # player too far from team centroid

# Known formations: name → list of line sizes (defence→attack)
KNOWN_FORMATIONS = {
    "4-4-2": [4, 4, 2],
    "4-3-3": [4, 3, 3],
    "4-2-3-1": [4, 2, 3, 1],
    "3-5-2": [3, 5, 2],
    "5-3-2": [5, 3, 2],
}

# Self-calibrating pitch model bounds
MIN_TEAM_WIDTH_OPEN_PLAY = 10.0
MAX_TEAM_WIDTH = 68.0


def _get_centre(bbox):
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _bbox_size(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def _px_to_world(cx, cy, ppm, vis_frac, frame_w=1920, frame_h=1080):
    """Convert pixel coords to world coords using calibration."""
    world_x = (cx / frame_w) * (PITCH_LENGTH * vis_frac)
    world_y = (cy / frame_h) * (PITCH_WIDTH * vis_frac)
    return world_x, world_y


def _clamp_to_pitch(wx, wy):
    """Clamp world position to pitch + margin."""
    wx = max(-PITCH_MARGIN, min(PITCH_LENGTH + PITCH_MARGIN, wx))
    wy = max(-PITCH_MARGIN, min(PITCH_WIDTH + PITCH_MARGIN, wy))
    return wx, wy


def _dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


class PhysicsCorrector:
    """Applies physical constraints to tracking data."""

    def __init__(self):
        self.stats = {
            "speed_corrections": 0,
            "position_corrections": 0,
            "collision_resolutions": 0,
            "trajectory_gaps_filled": 0,
            "homography_updates": 0,
            "tracks_merged_by_count": 0,
            "formation_outliers_flagged": 0,
            "final_pixels_per_metre": 0.0,
        }

    def apply_all_constraints(
        self,
        tracks: List[Dict],
        frame_metadata: List[Dict],
        calibration: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run all 7 constraints in order. Returns corrected tracks + report."""
        self._start_time = time.time()
        self._MAX_PHYSICS_SECONDS = 30

        ppm = calibration.get("pixels_per_metre", 15.5)
        vis_frac = calibration.get("visible_fraction", 0.55)

        # Constraint 1: normalise coordinates to pitch
        self._apply_pitch_normalisation(tracks, ppm, vis_frac)

        # Constraint 2: biomechanical limits
        self._apply_biomechanical_limits(tracks, ppm, vis_frac)

        # Constraint 5: temporal consistency + gap filling (before formation/collision)
        self._apply_temporal_consistency(tracks, ppm, vis_frac)

        # Constraint 6: collision detection
        self._apply_collision_detection(tracks, ppm, vis_frac, frame_metadata)

        # Constraint 4: player count enforcement
        self._apply_player_count_limit(tracks, ppm, vis_frac, frame_metadata)

        # Constraint 3: formation outlier detection
        self._apply_formation_validation(tracks, ppm, vis_frac, frame_metadata)

        # Constraint 7: self-calibrating pitch model
        ppm = self._apply_self_calibration(tracks, ppm, vis_frac, calibration)

        # Constraint 8: per-frame homography confidence — mark low-confidence frames
        self._apply_homography_confidence(tracks, calibration)

        # Constraint 9: pitch coverage gate — if coverage < 0.15,
        # mark ALL metrics as approximate (close-up / bench shot)
        pitch_coverage = calibration.get("pitch_coverage_score", 1.0)
        if pitch_coverage < 0.15:
            for track in tracks:
                for entry in track.get("trajectory", []):
                    entry["metric_quality"] = "approximate"
            self.stats["low_coverage_global_downgrade"] = True
            logger.warning(
                "Pitch coverage %.3f < 0.15 — all metrics "
                "marked approximate", pitch_coverage
            )

        self.stats["final_pixels_per_metre"] = round(ppm, 2)
        calibration["pixels_per_metre"] = ppm

        return {
            "corrected_tracks": tracks,
            "calibration": calibration,
            "corrections_applied": dict(self.stats),
        }

    # ── Constraint 1: Pitch normalisation ─────────────────────────

    def _apply_pitch_normalisation(self, tracks, ppm, vis_frac):
        """Map every detection to world coords, clamp to pitch boundaries."""
        for track in tracks:
            for entry in track.get("trajectory", []):
                cx, cy = _get_centre(entry["bbox"])
                wx, wy = _px_to_world(cx, cy, ppm, vis_frac)
                clamped_wx, clamped_wy = _clamp_to_pitch(wx, wy)
                if wx != clamped_wx or wy != clamped_wy:
                    self.stats["position_corrections"] += 1
                entry["world_x"] = round(clamped_wx, 2)
                entry["world_y"] = round(clamped_wy, 2)

    # ── Constraint 2: Biomechanical limits ────────────────────────

    def _apply_biomechanical_limits(self, tracks, ppm, vis_frac):
        """Cap speed, acceleration, and per-frame displacement."""
        for track in tracks:
            traj = track.get("trajectory", [])
            if len(traj) < 2:
                continue

            prev_speed = 0.0
            for i in range(1, len(traj)):
                prev = traj[i - 1]
                curr = traj[i]

                dt = curr.get("timestampSeconds", 0) - prev.get("timestampSeconds", 0)
                if dt <= 0:
                    continue

                wx1 = prev.get("world_x", 0)
                wy1 = prev.get("world_y", 0)
                wx2 = curr.get("world_x", 0)
                wy2 = curr.get("world_y", 0)

                dist_m = _dist((wx1, wy1), (wx2, wy2))
                speed = dist_m / dt

                # Max step per frame check
                frame_gap = max(1, abs(curr.get("frameIndex", 0) - prev.get("frameIndex", 0)))
                max_step = MAX_STEP_M * frame_gap
                if dist_m > max_step and speed > MAX_SPEED_MS:
                    # Interpolate: place at max plausible position along vector
                    ratio = max_step / dist_m if dist_m > 0 else 0
                    new_wx = wx1 + (wx2 - wx1) * ratio
                    new_wy = wy1 + (wy2 - wy1) * ratio
                    curr["world_x"] = round(new_wx, 2)
                    curr["world_y"] = round(new_wy, 2)
                    self.stats["speed_corrections"] += 1
                    speed = max_step / dt
                elif speed > MAX_SPEED_MS:
                    # Speed too high but step is small — cap speed by moving position
                    max_dist = MAX_SPEED_MS * dt
                    ratio = max_dist / dist_m if dist_m > 0 else 0
                    new_wx = wx1 + (wx2 - wx1) * ratio
                    new_wy = wy1 + (wy2 - wy1) * ratio
                    curr["world_x"] = round(new_wx, 2)
                    curr["world_y"] = round(new_wy, 2)
                    self.stats["speed_corrections"] += 1
                    speed = MAX_SPEED_MS

                # Acceleration check
                accel = (speed - prev_speed) / dt
                if accel > MAX_ACCEL_MS2:
                    # Smooth: limit speed increase
                    capped_speed = prev_speed + MAX_ACCEL_MS2 * dt
                    if dist_m > 0:
                        capped_dist = capped_speed * dt
                        ratio = capped_dist / dist_m
                        curr["world_x"] = round(wx1 + (wx2 - wx1) * ratio, 2)
                        curr["world_y"] = round(wy1 + (wy2 - wy1) * ratio, 2)
                        self.stats["speed_corrections"] += 1
                    speed = capped_speed
                elif accel < -MAX_DECEL_MS2:
                    capped_speed = max(0, prev_speed - MAX_DECEL_MS2 * dt)
                    if dist_m > 0:
                        capped_dist = capped_speed * dt
                        ratio = capped_dist / dist_m
                        curr["world_x"] = round(wx1 + (wx2 - wx1) * ratio, 2)
                        curr["world_y"] = round(wy1 + (wy2 - wy1) * ratio, 2)
                        self.stats["speed_corrections"] += 1
                    speed = capped_speed

                prev_speed = speed

    # ── Constraint 3: Formation validation ────────────────────────

    def _classify_formation(self, positions: List[Tuple[float, float]]) -> str:
        """Classify player positions into a known formation."""
        if len(positions) < 7:
            return "UNKNOWN"

        # Sort by x (depth on pitch) and cluster into lines
        sorted_pos = sorted(positions, key=lambda p: p[0])
        n = len(sorted_pos)

        # Try to fit known formations
        best_match = "UNKNOWN"
        best_score = float("inf")

        for name, lines in KNOWN_FORMATIONS.items():
            total_players = sum(lines)
            if abs(n - total_players) > 3:
                continue

            # Divide sorted positions into line groups
            boundaries = []
            cumulative = 0
            for line_size in lines:
                frac = line_size / total_players
                cumulative += frac
                boundaries.append(int(cumulative * n))

            # Score: variance of x within each line (lower = better fit)
            score = 0.0
            prev_b = 0
            for b in boundaries:
                group = sorted_pos[prev_b:b]
                if len(group) >= 2:
                    xs = [p[0] for p in group]
                    mean_x = sum(xs) / len(xs)
                    var = sum((x - mean_x) ** 2 for x in xs) / len(xs)
                    score += var
                prev_b = b

            if score < best_score:
                best_score = score
                best_match = name

        return best_match

    def _apply_formation_validation(self, tracks, ppm, vis_frac, frame_metadata):
        """Flag tracks whose positions are outliers relative to team centroid."""
        # Sample every ~2 seconds
        frame_indices = sorted(set(m.get("frameIndex", 0) for m in frame_metadata))
        sample_indices = frame_indices[::50] if len(frame_indices) > 50 else frame_indices[::max(1, len(frame_indices) // 5)]

        for frame_idx in sample_indices:
            for team_id in [0, 1]:
                team_positions = []
                team_tracks = []
                for t in tracks:
                    if t.get("teamId", -1) != team_id:
                        continue
                    if t.get("is_staff", False):
                        continue
                    traj = t.get("trajectory", [])
                    entry = min(
                        (e for e in traj if abs(e.get("frameIndex", 0) - frame_idx) <= 4),
                        key=lambda e: abs(e.get("frameIndex", 0) - frame_idx),
                        default=None,
                    )
                    if entry and "world_x" in entry:
                        team_positions.append((entry["world_x"], entry["world_y"]))
                        team_tracks.append((t, entry))

                if len(team_positions) < 4:
                    continue

                cx = sum(p[0] for p in team_positions) / len(team_positions)
                cy = sum(p[1] for p in team_positions) / len(team_positions)

                for (trk, entry), pos in zip(team_tracks, team_positions):
                    d = _dist(pos, (cx, cy))
                    if d > FORMATION_OUTLIER_M:
                        # Flag as low confidence for this frame
                        trk.setdefault("formation_outlier_frames", 0)
                        trk["formation_outlier_frames"] += 1
                        self.stats["formation_outliers_flagged"] += 1

    # ── Constraint 4: Player count enforcement ────────────────────

    def _apply_player_count_limit(self, tracks, ppm, vis_frac, frame_metadata):
        """If too many tracks active in a frame, merge closest pair."""
        frame_indices = sorted(set(m.get("frameIndex", 0) for m in frame_metadata))

        merged_ids = set()

        for frame_idx in frame_indices:
            if time.time() - self._start_time > self._MAX_PHYSICS_SECONDS:
                logger.warning("Physics corrector timeout after %.1fs — returning partial results", time.time() - self._start_time)
                break
            if len(merged_ids) > 20:
                break  # don't over-merge

            active = []
            for t in tracks:
                if t.get("trackId") in merged_ids:
                    continue
                if t.get("is_staff", False):
                    continue
                if (t.get("confirmed_detections", 0) or 0) < 3:
                    continue
                if t.get("firstSeen", 0) <= frame_idx <= t.get("lastSeen", 0):
                    traj = t.get("trajectory", [])
                    entry = min(
                        (e for e in traj if abs(e.get("frameIndex", 0) - frame_idx) <= 4),
                        key=lambda e: abs(e.get("frameIndex", 0) - frame_idx),
                        default=None,
                    )
                    if entry and "world_x" in entry:
                        active.append((t, entry))

            while len(active) > MAX_ACTIVE_TRACKS:
                # Find closest pair by position + same team
                best_dist = float("inf")
                best_pair = None
                for i in range(len(active)):
                    if time.time() - self._start_time > self._MAX_PHYSICS_SECONDS:
                        logger.warning("Physics corrector timeout after %.1fs — returning partial results", time.time() - self._start_time)
                        break
                    for j in range(i + 1, len(active)):
                        ti, ei = active[i]
                        tj, ej = active[j]
                        # Prefer merging same-team tracks
                        if ti.get("teamId", -1) != tj.get("teamId", -1):
                            continue
                        d = _dist(
                            (ei.get("world_x", 0), ei.get("world_y", 0)),
                            (ej.get("world_x", 0), ej.get("world_y", 0)),
                        )
                        if d < best_dist:
                            best_dist = d
                            best_pair = (i, j)

                if best_pair is None:
                    # No same-team pair, try any pair
                    for i in range(len(active)):
                        if time.time() - self._start_time > self._MAX_PHYSICS_SECONDS:
                            logger.warning("Physics corrector timeout after %.1fs — returning partial results", time.time() - self._start_time)
                            break
                        for j in range(i + 1, len(active)):
                            ei = active[i][1]
                            ej = active[j][1]
                            d = _dist(
                                (ei.get("world_x", 0), ei.get("world_y", 0)),
                                (ej.get("world_x", 0), ej.get("world_y", 0)),
                            )
                            if d < best_dist:
                                best_dist = d
                                best_pair = (i, j)

                if best_pair is None or best_dist > 10.0:
                    break  # don't merge distant tracks

                i, j = best_pair
                weak_idx = j if (active[i][0].get("confirmed_detections", 0) or 0) >= (active[j][0].get("confirmed_detections", 0) or 0) else i
                weak_track = active[weak_idx][0]
                weak_track["is_staff"] = True  # effectively remove it
                merged_ids.add(weak_track.get("trackId"))
                active.pop(weak_idx)
                self.stats["tracks_merged_by_count"] += 1

    # ── Constraint 5: Temporal consistency + gap filling ──────────

    def _apply_temporal_consistency(self, tracks, ppm, vis_frac):
        """Smooth trajectories and fill short gaps."""
        for track in tracks:
            traj = track.get("trajectory", [])
            if len(traj) < 3:
                continue

            # Sort by frame index
            traj.sort(key=lambda e: e.get("frameIndex", 0))

            # Fill gaps < GAP_FILL_MAX_S with linear interpolation
            filled = []
            for i in range(len(traj) - 1):
                filled.append(traj[i])
                curr = traj[i]
                nxt = traj[i + 1]

                t1 = curr.get("timestampSeconds", 0)
                t2 = nxt.get("timestampSeconds", 0)
                fi1 = curr.get("frameIndex", 0)
                fi2 = nxt.get("frameIndex", 0)
                gap = t2 - t1

                if 0.2 < gap <= GAP_FILL_MAX_S and fi2 - fi1 > 2:
                    # Interpolate intermediate points
                    wx1 = curr.get("world_x", 0)
                    wy1 = curr.get("world_y", 0)
                    wx2 = nxt.get("world_x", 0)
                    wy2 = nxt.get("world_y", 0)

                    # Check plausibility: max travel in gap
                    max_travel = MAX_SPEED_MS * gap
                    actual_travel = _dist((wx1, wy1), (wx2, wy2))
                    if actual_travel > max_travel:
                        continue  # skip — implausible gap

                    num_fill = min(int(gap * 25), 10)  # at ~25fps
                    for k in range(1, num_fill):
                        frac = k / num_fill
                        interp_fi = fi1 + int((fi2 - fi1) * frac)
                        interp_t = t1 + gap * frac
                        interp_wx = wx1 + (wx2 - wx1) * frac
                        interp_wy = wy1 + (wy2 - wy1) * frac

                        # Interpolate bbox too (linear)
                        b1 = curr["bbox"]
                        b2 = nxt["bbox"]
                        interp_bbox = [
                            b1[idx] + (b2[idx] - b1[idx]) * frac
                            for idx in range(4)
                        ]

                        filled.append({
                            "frameIndex": interp_fi,
                            "timestampSeconds": round(interp_t, 4),
                            "bbox": interp_bbox,
                            "world_x": round(interp_wx, 2),
                            "world_y": round(interp_wy, 2),
                            "interpolated": True,
                        })
                        self.stats["trajectory_gaps_filled"] += 1

            filled.append(traj[-1])

            # Deduplicate by frameIndex
            seen_frames = set()
            deduped = []
            for e in filled:
                fi = e.get("frameIndex", 0)
                if fi not in seen_frames:
                    seen_frames.add(fi)
                    deduped.append(e)

            deduped.sort(key=lambda e: e.get("frameIndex", 0))
            track["trajectory"] = deduped

    # ── Constraint 6: Collision detection ─────────────────────────

    def _apply_collision_detection(self, tracks, ppm, vis_frac, frame_metadata):
        """Resolve position overlaps between tracks in the same frame."""
        frame_indices = sorted(set(m.get("frameIndex", 0) for m in frame_metadata))

        # Sample every 5th frame for performance
        sampled = frame_indices[::5] if len(frame_indices) > 20 else frame_indices

        for frame_idx in sampled:
            if time.time() - self._start_time > self._MAX_PHYSICS_SECONDS:
                logger.warning("Physics corrector timeout after %.1fs — returning partial results", time.time() - self._start_time)
                break
            active = []
            for t in tracks:
                if t.get("is_staff", False):
                    continue
                traj = t.get("trajectory", [])
                entry = next(
                    (e for e in traj if abs(e.get("frameIndex", 0) - frame_idx) <= 2),
                    None,
                )
                if entry and "world_x" in entry:
                    conf = t.get("confirmed_detections", 0) or 0
                    active.append((t, entry, conf))

            # Check all pairs
            for i in range(len(active)):
                if time.time() - self._start_time > self._MAX_PHYSICS_SECONDS:
                    logger.warning("Physics corrector timeout after %.1fs — returning partial results", time.time() - self._start_time)
                    break
                for j in range(i + 1, len(active)):
                    ti, ei, ci = active[i]
                    tj, ej, cj = active[j]
                    d = _dist(
                        (ei.get("world_x", 0), ei.get("world_y", 0)),
                        (ej.get("world_x", 0), ej.get("world_y", 0)),
                    )
                    if d < COLLISION_DIST_M:
                        # Keep higher confidence, mark other as occluded
                        if ci >= cj:
                            # Push weaker track away slightly
                            ej["world_x"] = round(ej.get("world_x", 0) + COLLISION_DIST_M * 0.5, 2)
                        else:
                            ei["world_x"] = round(ei.get("world_x", 0) + COLLISION_DIST_M * 0.5, 2)
                        self.stats["collision_resolutions"] += 1

    # ── Constraint 8: Per-frame homography confidence ─────────────

    def _apply_homography_confidence(self, tracks, calibration):
        """
        Use frame_confidence_scores from calibration to mark trajectory entries
        that fall within low-confidence frames as approximate.

        Entries in low-confidence windows (score < 0.4) get:
          entry["metric_quality"] = "approximate"

        This is used downstream by velocity_service to know which speed/distance
        values should be treated as estimates rather than precise measurements.
        """
        frame_scores = calibration.get("frame_confidence_scores", [])
        if not frame_scores:
            return

        # Build a lookup: frame_index → confidence
        # Each score entry covers a window of 25 frames
        INTERVAL = 25
        low_conf_frames: set = set()
        for entry in frame_scores:
            if not entry.get("reliable", True):
                fi = entry.get("frame_index", 0)
                for offset in range(INTERVAL):
                    low_conf_frames.add(fi + offset)

        if not low_conf_frames:
            return

        low_conf_marked = 0
        for track in tracks:
            for traj_entry in track.get("trajectory", []):
                fi = traj_entry.get("frameIndex", 0)
                if fi in low_conf_frames:
                    traj_entry["metric_quality"] = "approximate"
                    low_conf_marked += 1

        self.stats.setdefault("low_conf_frames_marked", low_conf_marked)

    # ── Constraint 7: Self-calibrating pitch model ────────────────

    def _apply_self_calibration(self, tracks, ppm, vis_frac, calibration):
        """Use player point cloud to continuously refine pixels_per_metre."""
        # Collect all world positions grouped by frame
        frame_widths = []

        # Build frame → positions map from all tracks
        frame_positions: Dict[int, List[Tuple[float, float]]] = {}
        for t in tracks:
            if t.get("is_staff", False):
                continue
            if (t.get("confirmed_detections", 0) or 0) < 5:
                continue
            for entry in t.get("trajectory", []):
                if "world_x" not in entry:
                    continue
                fi = entry.get("frameIndex", 0)
                frame_positions.setdefault(fi, []).append(
                    (entry["world_x"], entry["world_y"])
                )

        # Sample every 5 frames
        sorted_frames = sorted(frame_positions.keys())
        sampled = sorted_frames[::5] if len(sorted_frames) > 20 else sorted_frames

        adjustments = []
        for fi in sampled:
            positions = frame_positions.get(fi, [])
            if len(positions) < 6:
                continue

            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            width = max(ys) - min(ys)  # lateral spread

            if width > MAX_TEAM_WIDTH:
                # Scale down: ppm is too low (positions too spread)
                scale = MAX_TEAM_WIDTH / width
                adjustments.append(scale)
                self.stats["homography_updates"] += 1
            elif width < MIN_TEAM_WIDTH_OPEN_PLAY and len(positions) >= 10:
                # Scale up: ppm is too high (positions too compact)
                scale = MIN_TEAM_WIDTH_OPEN_PLAY / width
                adjustments.append(scale)
                self.stats["homography_updates"] += 1
            else:
                adjustments.append(1.0)

        if adjustments:
            # Use median adjustment to be robust to outlier frames
            adjustments.sort()
            median_adj = adjustments[len(adjustments) // 2]

            # Apply conservatively — only adjust by up to 20%
            median_adj = max(0.80, min(1.20, median_adj))

            if abs(median_adj - 1.0) > 0.02:
                new_ppm = ppm * median_adj
                logger.info(
                    f"Self-calibration: ppm {ppm:.2f} → {new_ppm:.2f} "
                    f"(adjustment {median_adj:.3f})"
                )

                # Re-apply world coordinates with new ppm
                new_vis_frac = vis_frac  # keep vis_frac, adjust ppm
                for t in tracks:
                    for entry in t.get("trajectory", []):
                        cx, cy = _get_centre(entry["bbox"])
                        wx, wy = _px_to_world(cx, cy, new_ppm, new_vis_frac)
                        wx, wy = _clamp_to_pitch(wx, wy)
                        entry["world_x"] = round(wx, 2)
                        entry["world_y"] = round(wy, 2)

                return round(new_ppm, 2)

        return ppm
