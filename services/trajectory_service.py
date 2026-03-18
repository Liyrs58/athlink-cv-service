import json
import math
import os
import time
import urllib.request
from typing import Dict, List, Optional


class DevelopmentTrajectory:

    def compute_player_trajectory(
        self,
        player_appearances: list,
        player_id: str = "unknown",
    ) -> dict:
        """
        Takes a list of per-match performance snapshots for one
        player and computes their development trajectory.
        """
        # Filter to high/medium confidence only
        valid = [
            a for a in player_appearances
            if a.get("confidence") in ["high", "medium"]
        ]

        if len(valid) < 2:
            return {
                "player_id": player_id,
                "status": "insufficient_data",
                "appearances": len(player_appearances),
                "valid_appearances": len(valid),
                "message": "Need at least 2 high/medium confidence appearances",
            }

        # Sort by date
        valid.sort(key=lambda a: a.get("date", 0))

        # Extract metric series
        speeds = [a["max_speed_kmh"] for a in valid if a.get("max_speed_kmh")]
        sprints = [a["sprint_count"] for a in valid if a.get("sprint_count") is not None]
        distances = [a["distance_metres"] for a in valid if a.get("distance_metres")]
        fatigue_scores = [a["fatigue_score"] for a in valid if a.get("fatigue_score") is not None]

        # Compute linear trend for each metric
        speed_trend = self._linear_trend(speeds)
        sprint_trend = self._linear_trend(sprints)
        distance_trend = self._linear_trend(distances)
        fatigue_trend = self._linear_trend(fatigue_scores)

        # Predict next match values
        next_speed = self._predict_next(speeds, speed_trend)
        next_sprints = self._predict_next(sprints, sprint_trend)
        next_distance = self._predict_next(distances, distance_trend)

        # Development label
        dev_label = self._compute_development_label(
            speed_trend, sprint_trend, distance_trend, fatigue_trend
        )

        # Consistency score
        consistency = self._compute_consistency(speeds, sprints, distances)

        # Peak performance snapshot
        peak_speed = max(speeds) if speeds else None
        peak_sprints = max(sprints) if sprints else None
        peak_distance = max(distances) if distances else None

        # Generate Claude Haiku report if enough data
        report = None
        if len(valid) >= 3:
            report = self._generate_trajectory_report(
                player_id=player_id,
                appearances=len(valid),
                speed_trend=speed_trend,
                sprint_trend=sprint_trend,
                distance_trend=distance_trend,
                fatigue_trend=fatigue_trend,
                dev_label=dev_label,
                consistency=consistency,
                peak_speed=peak_speed,
                next_speed=next_speed,
                next_sprints=next_sprints,
            )

        return {
            "player_id": player_id,
            "status": "ok",
            "appearances_analysed": len(valid),
            "development_label": dev_label,
            "consistency_score": consistency,
            "trends": {
                "speed": speed_trend,
                "sprints": sprint_trend,
                "distance": distance_trend,
                "fatigue": fatigue_trend,
            },
            "current": {
                "avg_speed_kmh": round(sum(speeds) / len(speeds), 1) if speeds else None,
                "avg_sprints": round(sum(sprints) / len(sprints), 1) if sprints else None,
                "avg_distance_metres": round(sum(distances) / len(distances), 1) if distances else None,
            },
            "peak": {
                "max_speed_kmh": peak_speed,
                "max_sprints": peak_sprints,
                "max_distance_metres": peak_distance,
            },
            "predicted_next_match": {
                "speed_kmh": next_speed,
                "sprints": next_sprints,
                "distance_metres": next_distance,
            },
            "trajectory_report": report,
        }

    def _linear_trend(self, values: list) -> dict:
        """
        Computes linear regression slope on a series.
        Returns slope, direction, and strength.
        Pure Python — no numpy.
        """
        n = len(values)
        if n < 2:
            return {"slope": 0.0, "direction": "stable", "strength": 0.0}

        xs = list(range(n))
        mean_x = sum(xs) / n
        mean_y = sum(values) / n

        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values))
        denominator = sum((x - mean_x) ** 2 for x in xs)

        slope = numerator / denominator if denominator != 0 else 0.0

        # Normalise slope relative to mean value
        relative_slope = slope / mean_y if mean_y != 0 else 0.0

        if relative_slope > 0.05:
            direction = "improving"
        elif relative_slope < -0.05:
            direction = "declining"
        else:
            direction = "stable"

        # R² as strength of trend
        ss_res = sum(
            (y - (mean_y + slope * (x - mean_x))) ** 2
            for x, y in zip(xs, values)
        )
        ss_tot = sum((y - mean_y) ** 2 for y in values)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        strength = round(max(0.0, r_squared), 3)

        return {
            "slope": round(slope, 4),
            "direction": direction,
            "strength": strength,
            "n_points": n,
        }

    def _predict_next(self, values: list, trend: dict) -> Optional[float]:
        """
        Predicts the next value based on linear trend.
        Clamps to realistic bounds.
        """
        if not values or trend.get("slope") is None:
            return None
        last_val = values[-1]
        predicted = last_val + trend["slope"]
        return round(max(0.0, predicted), 1)

    def _compute_development_label(
        self,
        speed_trend: dict,
        sprint_trend: dict,
        distance_trend: dict,
        fatigue_trend: dict,
    ) -> str:
        """Overall development label based on metric trends."""
        improving = sum(
            1 for t in [speed_trend, sprint_trend, distance_trend]
            if t.get("direction") == "improving"
        )
        declining = sum(
            1 for t in [speed_trend, sprint_trend, distance_trend]
            if t.get("direction") == "declining"
        )

        fatigue_direction = fatigue_trend.get("direction", "stable")

        if improving >= 2 and fatigue_direction != "improving":
            return "EMERGING"
        elif improving >= 2 and fatigue_direction == "improving":
            return "BREAKTHROUGH"
        elif declining >= 2:
            return "DECLINING"
        elif improving == 1 and declining == 0:
            return "DEVELOPING"
        elif fatigue_direction == "declining":
            return "FATIGUING"
        else:
            return "PLATEAUING"

    def _compute_consistency(
        self,
        speeds: list,
        sprints: list,
        distances: list,
    ) -> float:
        """
        Consistency 0-100. Higher = more reliable performer.
        Based on coefficient of variation across metrics.
        """
        def cv(values):
            if len(values) < 2:
                return 1.0
            mean = sum(values) / len(values)
            if mean == 0:
                return 1.0
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std = variance ** 0.5
            return std / mean

        cvs = []
        if len(speeds) >= 2:
            cvs.append(cv(speeds))
        if len(sprints) >= 2:
            cvs.append(cv(sprints))
        if len(distances) >= 2:
            cvs.append(cv(distances))

        if not cvs:
            return 50.0

        avg_cv = sum(cvs) / len(cvs)
        consistency = max(0.0, min(100.0, (1 - avg_cv) * 100))
        return round(consistency, 1)

    def _generate_trajectory_report(
        self,
        player_id: str,
        appearances: int,
        speed_trend: dict,
        sprint_trend: dict,
        distance_trend: dict,
        fatigue_trend: dict,
        dev_label: str,
        consistency: float,
        peak_speed: float,
        next_speed: float,
        next_sprints: float,
    ) -> str:
        """
        Single Claude Haiku call to generate development report.
        Only called when 3+ appearances available.
        Uses urllib.request (no anthropic SDK).
        """
        prompt = f"""You are a youth football development analyst writing a
player trajectory report for a coach or academy scout.

Player ID: {player_id}
Matches analysed: {appearances}
Development label: {dev_label}
Consistency score: {consistency}/100

TREND DATA:
- Speed trend: {speed_trend.get("direction")} \
(slope: {speed_trend.get("slope")}, confidence: {speed_trend.get("strength")})
- Sprint trend: {sprint_trend.get("direction")} \
(slope: {sprint_trend.get("slope")}, confidence: {sprint_trend.get("strength")})
- Distance trend: {distance_trend.get("direction")} \
(slope: {distance_trend.get("slope")}, confidence: {distance_trend.get("strength")})
- Fatigue resilience: {fatigue_trend.get("direction")}

PEAK VALUES:
- Peak speed recorded: {peak_speed} km/h
- Predicted next match speed: {next_speed} km/h
- Predicted next match sprints: {next_sprints}

Write a Development Trajectory report with these exact sections:

## TRAJECTORY VERDICT
One sentence. What is the overall arc of this player's development?
Be direct. Do not hedge unless the data genuinely is unclear.

## WHAT THE DATA SHOWS
2-3 sentences on the specific trends. Reference the actual
numbers. Distinguish between improving and declining metrics.

## PHYSICAL CEILING ESTIMATE
1-2 sentences. Based on current trajectory, what is the
realistic physical ceiling for this player?
Only make this claim if trend strength > 0.5.
If strength is low, say the data is too early to project.

## DEVELOPMENT PRIORITY
2 specific, actionable things the coach should focus on
to accelerate this player's development based on the trends.

## SCOUT NOTE
One sentence a scout would write in a report.
Example: "Player shows consistent speed improvement across
3 matches — warrants closer monitoring."

Do not invent statistics. Do not use generic praise.
Write like someone who has actually watched the data."""

        data = json.dumps({
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 800,
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
            return f"Trajectory report unavailable: {str(e)}"

    def compute_team_trajectories(
        self,
        memory_data: dict,
        current_tracks: list,
        current_velocities: list,
        current_fatigue: dict,
    ) -> dict:
        """
        Entry point. Called from routes/analyse.py after each match.

        Extracts player appearances from memory, adds current match
        data, and computes trajectories for players with 2+ appearances.
        """
        # Get stored player history from memory
        player_history = memory_data.get("player_history", {})

        # Build current match appearance data per track
        vel_map = {v["track_id"]: v for v in current_velocities}
        fatigue_profiles = {
            p["track_id"]: p
            for p in current_fatigue.get("profiles", [])
        }

        current_appearances = {}
        now = time.time()

        for track in current_tracks:
            tid = track.get("trackId")
            if not tid:
                continue
            if track.get("is_staff", False):
                continue
            if (track.get("confirmed_detections", 0) or 0) < 5:
                continue

            vel = vel_map.get(tid, {})
            fatigue = fatigue_profiles.get(tid, {})

            current_appearances[str(tid)] = {
                "match_id": f"match_{int(now)}",
                "date": now,
                "max_speed_kmh": vel.get("max_speed_ms", 0) * 3.6,
                "sprint_count": vel.get("sprint_count", 0),
                "distance_metres": vel.get("distance_metres", 0),
                "fatigue_score": fatigue.get("fatigue_score", 50),
                "confidence": track.get("confidence_level", "low"),
            }

        # Merge with history
        for pid, appearance in current_appearances.items():
            player_history.setdefault(pid, []).append(appearance)

        # Compute trajectories for players with 2+ appearances
        trajectories = {}
        for pid, appearances in player_history.items():
            if len(appearances) >= 2:
                traj = self.compute_player_trajectory(appearances, player_id=pid)
                if traj.get("status") == "ok":
                    trajectories[pid] = traj

        return {
            "players_with_trajectory": len(trajectories),
            "trajectories": trajectories,
            "player_history_size": {
                pid: len(apps) for pid, apps in player_history.items()
            },
            "updated_player_history": player_history,
        }
