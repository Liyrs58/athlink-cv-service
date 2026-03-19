"""
Visual Intelligence Service — Claude Vision analysis of key frames
for tactical description and data auditing.
"""
import os
import json
import base64
import re
import logging
import urllib.request
import cv2

logger = logging.getLogger(__name__)


class VisualIntelligenceService:

    def extract_key_frames(self, video_path: str, job_id: str,
                           situation_events: list, num_frames: int = 8) -> list:
        """Extract frames at tactically important moments."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return []

            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            needs_rotation = raw_h > raw_w

            # Collect candidate timestamps with priorities
            candidates = []

            # Priority 1: First frame of each OPEN_PLAY phase
            for e in situation_events:
                if e.get("situation") == "OPEN_PLAY":
                    candidates.append({
                        "timestamp": e.get("start_time", 0),
                        "priority": 1,
                        "situation": "OPEN_PLAY",
                        "description": f"Open play start at {e.get('start_time', 0):.1f}s"
                    })

            # Priority 2: Phase transition points
            for i in range(1, len(situation_events)):
                prev = situation_events[i - 1]
                curr = situation_events[i]
                if prev.get("situation") != curr.get("situation"):
                    ts = curr.get("start_time", 0)
                    candidates.append({
                        "timestamp": ts,
                        "priority": 2,
                        "situation": curr.get("situation", "UNKNOWN"),
                        "description": f"Phase transition at {ts:.1f}s"
                    })

            # Priority 3: Middle of longest OPEN_PLAY phase
            open_play_phases = [e for e in situation_events if e.get("situation") == "OPEN_PLAY"]
            if open_play_phases:
                longest = max(open_play_phases,
                              key=lambda e: (e.get("end_time", 0) - e.get("start_time", 0)))
                mid_ts = (longest.get("start_time", 0) + longest.get("end_time", 0)) / 2.0
                candidates.append({
                    "timestamp": mid_ts,
                    "priority": 3,
                    "situation": "OPEN_PLAY",
                    "description": f"Mid open play at {mid_ts:.1f}s"
                })

            # Sort by priority, deduplicate by proximity (within 1s)
            candidates.sort(key=lambda c: c["priority"])
            selected = []
            used_timestamps = set()
            for c in candidates:
                ts = round(c["timestamp"], 1)
                too_close = any(abs(ts - u) < 1.0 for u in used_timestamps)
                if not too_close and len(selected) < num_frames:
                    selected.append(c)
                    used_timestamps.add(ts)

            # Priority 4: Fill remainder evenly
            if len(selected) < num_frames and duration > 0:
                remaining = num_frames - len(selected)
                for i in range(remaining):
                    ts = round(duration * (i + 1) / (remaining + 1), 1)
                    too_close = any(abs(ts - u) < 1.0 for u in used_timestamps)
                    if not too_close:
                        selected.append({
                            "timestamp": ts,
                            "priority": 4,
                            "situation": "FILL",
                            "description": f"Evenly sampled at {ts:.1f}s"
                        })
                        used_timestamps.add(ts)

            # Sort by timestamp for sequential reading
            selected.sort(key=lambda c: c["timestamp"])

            # Extract frames
            frames = []
            for candidate in selected:
                ts = candidate["timestamp"]
                frame_idx = int(ts * fps)
                frame_idx = max(0, min(frame_idx, total_frames - 1))

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                if needs_rotation:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                # Resize to max 1280px wide
                h, w = frame.shape[:2]
                if w > 1280:
                    scale = 1280 / w
                    new_w = 1280
                    new_h = int(h * scale)
                    frame = cv2.resize(frame, (new_w, new_h))

                # Save as JPEG
                frame_path = f"/tmp/{job_id}_frame_{ts:.1f}s.jpg"
                cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

                frames.append({
                    "timestamp": ts,
                    "situation": candidate["situation"],
                    "frame_path": frame_path,
                    "description": candidate["description"],
                })

            cap.release()
            return frames

        except Exception as e:
            logger.warning(f"Visual Intelligence frame extraction failed: {e}")
            return []

    def analyse_and_correct(self, video_path: str, job_id: str,
                             situation_events: list, tracking_stats: dict,
                             physical_players: list, brain_summary: dict) -> dict:
        """Two sequential Claude Vision calls: tactical description + data audit."""

        frames = self.extract_key_frames(video_path, job_id, situation_events)

        if not frames:
            return {
                "tactical_narrative": None,
                "key_moments": [],
                "vision_confidence": "unavailable",
                "frames_analysed": 0,
                "data_flags": [],
                "corrections_applied": 0,
                "corrected_players": physical_players,
                "audit_summary": "Frame extraction failed"
            }

        tactical_narrative = None
        key_moments = []
        try:
            tactical_prompt = self._build_tactical_prompt(frames, tracking_stats, brain_summary)
            tactical_response = self._call_vision(frames, tactical_prompt)
            tactical_narrative, key_moments = self._parse_tactical(tactical_response)
        except Exception as e:
            logger.warning(f"Visual Intelligence tactical call failed: {e}")

        data_flags = []
        audit_summary = "Audit unavailable"
        try:
            audit_prompt = self._build_audit_prompt(frames, physical_players, situation_events, tracking_stats)
            audit_response = self._call_vision(frames, audit_prompt)
            data_flags, audit_summary = self._parse_audit(audit_response)
        except Exception as e:
            logger.warning(f"Visual Intelligence audit call failed: {e}")

        # Apply only high confidence flags
        corrected_players, num_applied, apply_summary = self._apply_flags(physical_players, data_flags)
        if apply_summary:
            audit_summary = apply_summary

        # Cleanup temp frames
        for f in frames:
            try:
                os.remove(f["frame_path"])
            except Exception:
                pass

        # Vision confidence
        reliability = brain_summary.get("tracking_health", {}).get("data_reliability", "low")
        if len(frames) >= 6 and reliability == "high":
            vision_conf = "high"
        elif len(frames) >= 4:
            vision_conf = "medium"
        else:
            vision_conf = "low"

        return {
            "tactical_narrative": tactical_narrative,
            "key_moments": key_moments,
            "vision_confidence": vision_conf,
            "frames_analysed": len(frames),
            "data_flags": data_flags,
            "corrections_applied": num_applied,
            "corrected_players": corrected_players,
            "audit_summary": audit_summary,
        }

    def _call_vision(self, frames: list, prompt: str) -> str:
        """Claude Vision API call using urllib.request — same pattern as interpretation_service.py."""
        content = []

        for frame in frames:
            frame_path = frame["frame_path"]
            try:
                with open(frame_path, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode("utf-8")
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_data,
                    }
                })
                content.append({
                    "type": "text",
                    "text": f"[Frame at {frame['timestamp']:.1f}s — {frame['description']}]"
                })
            except Exception:
                continue

        content.append({"type": "text", "text": prompt})

        data = json.dumps({
            "model": "claude-sonnet-4-6-20250514",
            "max_tokens": 1500,
            "messages": [{"role": "user", "content": content}]
        }).encode()

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=data,
            headers={
                "Content-Type": "application/json",
                "x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""),
                "anthropic-version": "2023-06-01",
            }
        )

        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
            return result["content"][0]["text"]

    def _build_tactical_prompt(self, frames: list, tracking_stats: dict,
                                brain_summary: dict) -> str:
        t0_name = tracking_stats.get("team_0_colour_name", "Team A")
        t1_name = tracking_stats.get("team_1_colour_name", "Team B")
        t0_players = tracking_stats.get("team_0_players", "?")
        t1_players = tracking_stats.get("team_1_players", "?")
        total_sprints = tracking_stats.get("total_sprints", "?")
        max_speed = tracking_stats.get("max_speed_kmh", "?")
        clip_duration = tracking_stats.get("clip_duration", "?")
        reliability = brain_summary.get("tracking_health", {}).get("data_reliability", "unknown")

        # Phase timeline
        phase_lines = []
        events = brain_summary.get("match_phases", [])
        for p in events:
            phase_lines.append(f"  {p.get('phase', '?')} {p.get('start', '?')}s–{p.get('end', '?')}s")
        phase_text = "\n".join(phase_lines) if phase_lines else "  unavailable"

        return f"""You are watching {len(frames)} frames from a football clip.

Teams: {t0_name} ({t0_players} players) vs {t1_name} ({t1_players} players)
Tracking data: {total_sprints} total sprints, {max_speed} km/h max speed
Brain reliability: {reliability}
Clip duration: {clip_duration} seconds
Phase timeline:
{phase_text}

Describe ONLY what you can see. Do not invent events you cannot see in these frames.

BANNED WORDS: entropy, Voronoi, homography, pixel, heuristic, algorithm, belief state, pressing triggers, half-space, gegenpressing, progressive carries.

BANNED TACTICS: Do not write "high press", "low block", or any formation number (4-3-3 etc) unless you can literally count the players in those positions across multiple frames.

Write EXACTLY these sections, no others:

WHAT HAPPENED:
2-3 sentences. The story of the clip in order. Use actual timestamps from the frames.

KEY MOMENT:
One sentence. The single most important thing that happened. Must reference a specific timestamp.

TEAM SHAPE:
One sentence per team. Describe what you see — wide/compact, numbers in defence vs attack. Do not name formations.

INTENSITY CHECK:
The tracking data says {total_sprints} sprints and {max_speed} km/h max speed. Does what you see in the frames support this? Say yes, roughly yes, or flag if it looks wrong."""

    def _build_audit_prompt(self, frames: list, physical_players: list,
                             situation_events: list, tracking_stats: dict) -> str:
        # Build player table
        table_lines = ["Track | Team | Sprints | Distance | MaxSpeed | Confidence"]
        for p in physical_players:
            conf = p.get("confidence", "low")
            if conf not in ("high", "medium"):
                continue
            dist = p.get("distance_metres", 0)
            if dist < 5:
                continue
            track_id = p.get("track_id", "?")
            team = p.get("team", "?")
            sprints = p.get("sprints", 0)
            max_spd = p.get("max_speed_kmh", 0)
            table_lines.append(
                f"#{track_id} | {team} | {sprints} | {dist:.0f}m | {max_spd}kmh | {conf}"
            )

        player_table = "\n".join(table_lines)
        clip_duration = tracking_stats.get("clip_duration", "?")

        return f"""You are auditing computer vision tracking data against what you can see in these frames.

Player data:
{player_table}

Clip duration: {clip_duration} seconds
Frames shown: {len(frames)} out of full clip

Physical limits to apply:
- Max human sprint speed: 36 km/h
- Sustained average over full clip above 18 km/h is physically impossible
- A player visible as stationary in 3+ frames likely did not cover large distances
- Sprint count above 8 in a 40-second clip is extremely unlikely

Flag a correction ONLY when you are genuinely confident something is wrong.
Do not flag plausible values.
Do not flag values just because they seem high — only flag when they contradict physics or what you can see.

Respond ONLY in valid JSON, no markdown fences:
{{"flags": [{{"track_id": 1837, "metric": "distance", "original_value": 176, "suggested_value": 95, "confidence": "high", "reason": "176m in 40s = 15.8 km/h sustained average. Player appears stationary in frames 3 and 6."}}], "audit_summary": "One distance value appears inflated based on visible player movement. Sprint counts look accurate."}}

If nothing looks wrong:
{{"flags": [], "audit_summary": "All values look plausible."}}"""

    def _parse_tactical(self, response: str) -> tuple:
        """Parse Call 1 response into narrative and key moments."""
        try:
            narrative = response.strip() if response else ""
            key_moments = []

            # Extract KEY MOMENT section timestamp
            km_match = re.search(r'KEY MOMENT[:\s]*\n?(.*?)(?:\n\n|\nTEAM|\Z)', response, re.DOTALL | re.IGNORECASE)
            if km_match:
                km_text = km_match.group(1).strip()
                # Find timestamp like "12.4s" or "at 12.4s"
                ts_match = re.search(r'(\d+\.?\d*)\s*s', km_text)
                if ts_match:
                    key_moments.append({
                        "timestamp": float(ts_match.group(1)),
                        "description": km_text,
                    })

            return narrative, key_moments
        except Exception:
            return "", []

    def _parse_audit(self, response: str) -> tuple:
        """Parse Call 2 JSON response. Returns (flags_list, audit_summary)."""
        try:
            # Strip markdown fences
            text = response.strip()
            text = re.sub(r'^```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)

            parsed = json.loads(text)
            flags = parsed.get("flags", [])
            summary = parsed.get("audit_summary", "Audit complete")

            # Normalize flag format
            normalized = []
            for f in flags:
                normalized.append({
                    "track_id": f.get("track_id"),
                    "metric": f.get("metric", "unknown"),
                    "original_value": f.get("original_value"),
                    "suggested_value": f.get("suggested_value"),
                    "flag_confidence": f.get("confidence", "medium"),
                    "reason": f.get("reason", ""),
                    "applied": False,  # will be set by _apply_flags
                })

            return normalized, summary
        except Exception:
            return [], "Audit parse failed"

    def _apply_flags(self, physical_players: list, data_flags: list) -> tuple:
        """Apply high confidence flags to player list."""
        import copy
        corrected_players = copy.deepcopy(physical_players)
        num_applied = 0

        for flag in data_flags:
            if flag.get("flag_confidence") != "high":
                continue

            track_id = flag.get("track_id")
            metric = flag.get("metric")
            suggested = flag.get("suggested_value")

            if track_id is None or metric is None or suggested is None:
                continue

            # Find matching player
            for player in corrected_players:
                if player.get("track_id") == track_id:
                    # Map metric names to player dict keys
                    key_map = {
                        "distance": "distance_metres",
                        "distance_metres": "distance_metres",
                        "max_speed": "max_speed_kmh",
                        "max_speed_kmh": "max_speed_kmh",
                        "sprints": "sprints",
                        "sprint_count": "sprints",
                    }
                    player_key = key_map.get(metric, metric)

                    if player_key in player:
                        # Preserve original
                        player[f"original_{metric}"] = player[player_key]
                        player[player_key] = suggested
                        player["vision_corrected"] = True
                        flag["applied"] = True
                        num_applied += 1
                    break

        summary_parts = []
        if num_applied > 0:
            summary_parts.append(f"{num_applied} correction(s) applied")
        medium_count = sum(1 for f in data_flags if f.get("flag_confidence") == "medium")
        if medium_count > 0:
            summary_parts.append(f"{medium_count} warning(s) logged")
        if not summary_parts:
            summary_parts.append("No corrections needed")

        return corrected_players, num_applied, ". ".join(summary_parts)
