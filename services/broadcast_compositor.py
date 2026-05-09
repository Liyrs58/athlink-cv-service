"""broadcast_compositor.py — 2× supersampled overlay layer + label layout solver.

Usage:
    comp = OverlayCompositor(w=1920, h=1080, scale=2)
    comp.draw_circle((cx, cy), radius, color=(0,255,0), thickness=3)
    comp.draw_text("BALL CARRIER", (cx, cy-40), font_scale=1.0, color=(255,255,255))
    result = comp.composite(base_frame)  # returns (h, w, 3) uint8 BGR

The compositor draws everything at 2× (or 3×) resolution using BGRA pixel space,
then Lanczos-downscales and alpha-composites onto the base frame. This gives
significantly cleaner edges on text, circles, and polylines at broadcast resolution.

Label layout solver (`solve_label_positions`) provides simple collision-avoidance:
tries above, above-left, above-right, side positions in priority order and picks
the first one that doesn't overlap already-placed labels or the scoreboard zone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ── Layer stack order ──────────────────────────────────────────────────────
LAYER_ZONE_FILL = 0
LAYER_GLOW      = 1
LAYER_LINES     = 2
LAYER_RINGS     = 3
LAYER_LABELS    = 4
LAYER_TITLES    = 5
_N_LAYERS       = 6


class OverlayCompositor:
    """Thread-unsafe; create one per frame (cheap allocation)."""

    def __init__(self, w: int, h: int, scale: int = 2) -> None:
        self.w = w
        self.h = h
        self.scale = scale
        # One BGRA layer per rendering pass, all transparent (alpha=0) initially.
        self._layers: List[np.ndarray] = [
            np.zeros((h * scale, w * scale, 4), dtype=np.uint8)
            for _ in range(_N_LAYERS)
        ]

    # ── Internal helpers ───────────────────────────────────────────────────

    def _s(self, xy: Tuple[int, int]) -> Tuple[int, int]:
        """Scale a pixel coordinate to overlay resolution."""
        return (int(xy[0] * self.scale), int(xy[1] * self.scale))

    def _layer(self, layer_idx: int) -> np.ndarray:
        return self._layers[layer_idx]

    @staticmethod
    def _bgra(color_bgr: Tuple[int, int, int], alpha: int = 255) -> Tuple[int, int, int, int]:
        return (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]), alpha)

    # ── Drawing API ────────────────────────────────────────────────────────

    def draw_glow_line(
        self,
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        color: Tuple[int, int, int],
        *,
        thickness: int = 10,
        glow_radius: int = 18,
        glow_alpha: float = 0.45,
        layer: int = LAYER_GLOW,
    ) -> None:
        """Semi-transparent glow halo + bright core line."""
        sp1, sp2 = self._s(p1), self._s(p2)
        t_s = thickness * self.scale
        g_s = glow_radius * self.scale
        lyr = self._layer(layer)
        # Glow: draw on a temp layer, Gaussian blur, write back
        glow_tmp = np.zeros_like(lyr[:, :, :3])
        cv2.line(glow_tmp, sp1, sp2, color, t_s + g_s * 2, cv2.LINE_AA)
        ksize = max(3, g_s | 1)  # must be odd
        blurred = cv2.GaussianBlur(glow_tmp, (ksize, ksize), 0)
        alpha_mask = (blurred.max(axis=2) * glow_alpha).astype(np.uint8)
        # Composite glow into layer
        glow_bgra = np.dstack([blurred, alpha_mask])
        _alpha_blend_bgra(lyr, glow_bgra)
        # Core line (fully opaque)
        cv2.line(lyr, sp1, sp2, self._bgra(color, 255), t_s, cv2.LINE_AA)

    def draw_circle(
        self,
        center: Tuple[int, int],
        radius: int,
        color: Tuple[int, int, int],
        *,
        thickness: int = 3,
        alpha: int = 255,
        layer: int = LAYER_RINGS,
    ) -> None:
        sc = self._s(center)
        r_s = radius * self.scale
        t_s = thickness * self.scale
        cv2.circle(self._layer(layer), sc, r_s, self._bgra(color, alpha), t_s, cv2.LINE_AA)

    def fill_polygon(
        self,
        points: List[Tuple[int, int]],
        color: Tuple[int, int, int],
        *,
        alpha: int = 60,
        layer: int = LAYER_ZONE_FILL,
    ) -> None:
        if len(points) < 3:
            return
        pts = np.array([self._s(p) for p in points], dtype=np.int32)
        hull = cv2.convexHull(pts)
        fill_layer = np.zeros_like(self._layer(layer))
        cv2.fillPoly(fill_layer, [hull], self._bgra(color, alpha), cv2.LINE_AA)
        _alpha_blend_bgra(self._layer(layer), fill_layer)
        # Outline
        cv2.polylines(self._layer(layer), [hull], True, self._bgra(color, 210), 3 * self.scale, cv2.LINE_AA)

    def fill_circle(
        self,
        center: Tuple[int, int],
        radius: int,
        color: Tuple[int, int, int],
        *,
        alpha: int = 255,
        layer: int = LAYER_RINGS,
    ) -> None:
        sc = self._s(center)
        r_s = max(1, radius * self.scale)
        cv2.circle(self._layer(layer), sc, r_s, self._bgra(color, alpha), -1, cv2.LINE_AA)

    def fill_ellipse(
        self,
        center: Tuple[int, int],
        axes: Tuple[int, int],
        color: Tuple[int, int, int],
        *,
        alpha: int = 255,
        layer: int = LAYER_RINGS,
    ) -> None:
        sc = self._s(center)
        sa = (max(1, axes[0] * self.scale), max(1, axes[1] * self.scale))
        cv2.ellipse(self._layer(layer), sc, sa, 0, 0, 360,
                    self._bgra(color, alpha), -1, cv2.LINE_AA)

    def draw_curved_arrow(
        self,
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        color: Tuple[int, int, int],
        *,
        thickness: int = 4,
        bend: float = 0.30,
        alpha: int = 255,
        stop_short_px: int = 32,
        layer: int = LAYER_LINES,
    ) -> None:
        """Quadratic-Bezier arrow with shadow halo + arrowhead."""
        sp1, sp2 = self._s(p1), self._s(p2)
        t_s = max(1, thickness * self.scale)
        stop_s = stop_short_px * self.scale
        # Trim end to leave headroom around target ring
        dx, dy = sp2[0] - sp1[0], sp2[1] - sp1[1]
        norm = (dx * dx + dy * dy) ** 0.5
        if norm > stop_s + 4:
            f = (norm - stop_s) / norm
            sp2 = (int(sp1[0] + dx * f), int(sp1[1] + dy * f))
        # Bezier control: perpendicular offset by bend * length
        mx, my = (sp1[0] + sp2[0]) // 2, (sp1[1] + sp2[1]) // 2
        nx, ny = -dy / max(1, norm), dx / max(1, norm)
        cx = int(mx + nx * bend * norm)
        cy = int(my + ny * bend * norm)
        # Sample 24 points
        pts: List[Tuple[int, int]] = []
        for i in range(25):
            t = i / 24.0
            u = 1.0 - t
            bx = int(u * u * sp1[0] + 2 * u * t * cx + t * t * sp2[0])
            by = int(u * u * sp1[1] + 2 * u * t * cy + t * t * sp2[1])
            pts.append((bx, by))
        lyr = self._layer(layer)
        # Shadow first (darker, thicker)
        for i in range(len(pts) - 1):
            cv2.line(lyr, pts[i], pts[i + 1], (0, 0, 0, 180),
                     t_s + 2 * self.scale, cv2.LINE_AA)
        # Bright core
        for i in range(len(pts) - 1):
            cv2.line(lyr, pts[i], pts[i + 1], self._bgra(color, alpha),
                     t_s, cv2.LINE_AA)
        # Arrowhead
        tip = pts[-1]
        before = pts[-3]
        ang = np.arctan2(tip[1] - before[1], tip[0] - before[0])
        head_len = 12 * self.scale
        head_ang = 0.45  # rad
        a1 = (int(tip[0] - head_len * np.cos(ang - head_ang)),
              int(tip[1] - head_len * np.sin(ang - head_ang)))
        a2 = (int(tip[0] - head_len * np.cos(ang + head_ang)),
              int(tip[1] - head_len * np.sin(ang + head_ang)))
        cv2.line(lyr, tip, a1, self._bgra(color, alpha), t_s, cv2.LINE_AA)
        cv2.line(lyr, tip, a2, self._bgra(color, alpha), t_s, cv2.LINE_AA)

    def draw_pill(
        self,
        center: Tuple[int, int],
        text: str,
        color: Tuple[int, int, int],
        *,
        font_scale: float = 0.48,
        alpha: int = 230,
        layer: int = LAYER_LABELS,
    ) -> None:
        """Solid coloured pill with white text, supersampled."""
        if not text:
            return
        sp = self._s(center)
        fs = font_scale * self.scale
        th = max(1, int(round(self.scale)))
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, hh), _ = cv2.getTextSize(text.upper(), font, fs, th)
        pad_x, pad_y = 10 * self.scale, 6 * self.scale
        bw, bh = tw + 2 * pad_x, hh + 2 * pad_y
        x1, y1 = sp[0] - bw // 2, sp[1] - bh
        x2, y2 = sp[0] + bw // 2, sp[1]
        lyr = self._layer(layer)
        # Shadow
        cv2.rectangle(lyr, (x1 + self.scale, y1 + self.scale),
                      (x2 + self.scale, y2 + self.scale),
                      (0, 0, 0, 160), -1)
        # Pill body
        cv2.rectangle(lyr, (x1, y1), (x2, y2), self._bgra(color, alpha), -1)
        # Text — black halo behind for legibility
        tx = x1 + pad_x
        ty = y2 - pad_y
        cv2.putText(lyr, text.upper(), (tx + 1, ty + 1), font, fs,
                    (0, 0, 0, 220), th + self.scale, cv2.LINE_AA)
        cv2.putText(lyr, text.upper(), (tx, ty), font, fs,
                    (255, 255, 255, 255), th, cv2.LINE_AA)

    def draw_text(
        self,
        text: str,
        pos: Tuple[int, int],
        *,
        font_scale: float = 0.9,
        color: Tuple[int, int, int] = (255, 255, 255),
        thickness: int = 2,
        layer: int = LAYER_LABELS,
        shadow: bool = True,
    ) -> None:
        sp = self._s(pos)
        fs = font_scale * self.scale
        th = thickness * self.scale
        font = cv2.FONT_HERSHEY_SIMPLEX
        if shadow:
            off = self.scale
            cv2.putText(self._layer(layer), text, (sp[0] + off, sp[1] + off),
                        font, fs, (0, 0, 0, 220), th + self.scale, cv2.LINE_AA)
        cv2.putText(self._layer(layer), text, sp, font, fs, self._bgra(color, 255), th, cv2.LINE_AA)

    # ── Compositing ────────────────────────────────────────────────────────

    def composite(self, base_frame: np.ndarray) -> np.ndarray:
        """Merge all layers onto base_frame (H×W×3 BGR). Returns new frame."""
        result = base_frame.copy()
        for lyr in self._layers:
            if lyr.max() == 0:
                continue  # empty layer, skip
            # Downscale to target resolution with Lanczos
            lyr_small = cv2.resize(lyr, (self.w, self.h), interpolation=cv2.INTER_LANCZOS4)
            # Alpha-blend onto result
            alpha_ch = lyr_small[:, :, 3:4].astype(np.float32) / 255.0
            bgr = lyr_small[:, :, :3].astype(np.float32)
            result = (result.astype(np.float32) * (1.0 - alpha_ch) + bgr * alpha_ch).astype(np.uint8)
        return result


def _alpha_blend_bgra(dst: np.ndarray, src: np.ndarray) -> None:
    """Blend src BGRA over dst BGRA in-place (straight alpha)."""
    a = src[:, :, 3:4].astype(np.float32) / 255.0
    dst[:, :, :3] = (dst[:, :, :3].astype(np.float32) * (1.0 - a) + src[:, :, :3].astype(np.float32) * a).astype(np.uint8)
    dst[:, :, 3] = np.maximum(dst[:, :, 3], src[:, :, 3])


# ── Label layout solver ────────────────────────────────────────────────────

@dataclass
class LabelSpec:
    anchor: Tuple[int, int]        # foot/ring pixel position
    text: str
    priority: int = 0              # higher = gets position first
    ring_radius: int = 28
    color: Tuple[int, int, int] = (255, 255, 255)
    # Resolved after solve_label_positions:
    resolved_pos: Optional[Tuple[int, int]] = field(default=None, repr=False)
    resolved_side: str = "above"


# No-fly zones: top-left scoreboard chip area
_SCOREBOARD_EXCLUSION = (0, 0, 320, 110)  # (x1, y1, x2, y2)


def _aabb(pos: Tuple[int, int], text: str, font_scale: float = 0.48) -> Tuple[int, int, int, int]:
    """Return (x1, y1, x2, y2) for a text pill at pos."""
    (tw, hh), _ = cv2.getTextSize(text.upper(), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    pad_x, pad_y = 10, 6
    bw = tw + 2 * pad_x
    bh = hh + 2 * pad_y
    cx, cy = pos
    return (cx - bw // 2, cy - bh, cx + bw // 2, cy)


def _overlaps(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int], margin: int = 6) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 + margin < bx1 or bx2 + margin < ax1 or ay2 + margin < by1 or by2 + margin < ay1)


def _candidate_positions(spec: LabelSpec, frame_w: int, frame_h: int) -> List[Tuple[Tuple[int, int], str]]:
    """Yield (pos, side_name) candidates in priority order."""
    ax, ay = spec.anchor
    r = spec.ring_radius
    offset = r + 14
    candidates = [
        ((ax, ay - offset),           "above"),
        ((ax - offset, ay - offset),  "above_left"),
        ((ax + offset, ay - offset),  "above_right"),
        ((ax + offset, ay),           "right"),
        ((ax - offset, ay),           "left"),
        ((ax, ay + offset),           "below"),
    ]
    return [(pos, side) for pos, side in candidates]


def solve_label_positions(
    labels: List[LabelSpec],
    frame_w: int,
    frame_h: int,
) -> List[LabelSpec]:
    """Assign non-overlapping positions to all labels.

    Modifies each spec in-place (resolved_pos, resolved_side) and returns the list.
    Higher-priority labels claim their preferred position first.
    """
    sorted_specs = sorted(labels, key=lambda s: -s.priority)
    placed: List[Tuple[int, int, int, int]] = [_SCOREBOARD_EXCLUSION]

    for spec in sorted_specs:
        chosen_pos = None
        chosen_side = "above"

        for pos, side in _candidate_positions(spec, frame_w, frame_h):
            box = _aabb(pos, spec.text)
            # Clamp inside frame
            bx1, by1, bx2, by2 = box
            if bx1 < 4 or by1 < 4 or bx2 > frame_w - 4 or by2 > frame_h - 4:
                continue
            if any(_overlaps(box, p) for p in placed):
                continue
            chosen_pos = pos
            chosen_side = side
            placed.append(box)
            break

        if chosen_pos is None:
            # No valid position found — fall back to above-anchor (may overlap)
            chosen_pos = (spec.anchor[0], spec.anchor[1] - spec.ring_radius - 14)
            chosen_side = "above_fallback"

        spec.resolved_pos = chosen_pos
        spec.resolved_side = chosen_side

    return sorted_specs
