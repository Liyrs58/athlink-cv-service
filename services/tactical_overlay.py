"""Drawing primitives for the Performance Zone renderer.

Pure cv2 helpers — no state, no I/O. Pulled out of render_performance_zone.py
so the renderer file stays under control as we add tactical layers.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np


# ── arrows ────────────────────────────────────────────────────────────────

def draw_arrow(
    img: np.ndarray,
    start_px: Tuple[int, int],
    end_px: Tuple[int, int],
    color: Tuple[int, int, int],
    *,
    thickness: int = 3,
    head_len: int = 14,
    glow: bool = True,
) -> None:
    """Straight arrow from start to end. Optional thin black halo for legibility on grass."""
    if start_px is None or end_px is None:
        return
    if glow:
        cv2.line(img, start_px, end_px, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.arrowedLine(
        img, start_px, end_px, color, thickness,
        tipLength=max(0.05, min(0.4, head_len / max(1.0, np.hypot(end_px[0] - start_px[0], end_px[1] - start_px[1])))),
        line_type=cv2.LINE_AA,
    )


def draw_curved_arrow(
    img: np.ndarray,
    start_px: Tuple[int, int],
    end_px: Tuple[int, int],
    color: Tuple[int, int, int],
    *,
    thickness: int = 4,
    bend: float = 0.25,
    stop_short_px: int = 28,
    glow: bool = True,
) -> None:
    """Quadratic-Bezier arrow from start to end with an inward bend.
    Reads as a closing-pressure motion, not a straight pass line.
    `bend` is signed — positive bows clockwise, negative anti-clockwise.
    `stop_short_px` shortens the head so it lands outside the target ring."""
    if start_px is None or end_px is None:
        return
    sx, sy = start_px
    ex, ey = end_px
    dx, dy = ex - sx, ey - sy
    length = float(np.hypot(dx, dy))
    if length < 4:
        return

    # Trim end-point so the head doesn't crash into the target ring
    trim = min(stop_short_px, int(length * 0.4))
    ex2 = int(ex - dx * (trim / length))
    ey2 = int(ey - dy * (trim / length))

    # Control point: midpoint pushed perpendicularly by `bend` * length
    mx, my = (sx + ex2) / 2.0, (sy + ey2) / 2.0
    nx, ny = -(ey2 - sy) / max(1.0, length), (ex2 - sx) / max(1.0, length)
    cx = int(mx + nx * length * bend)
    cy = int(my + ny * length * bend)

    # Sample 24 points along the Bezier
    pts = []
    for i in range(25):
        t = i / 24.0
        u = 1.0 - t
        bx = u * u * sx + 2 * u * t * cx + t * t * ex2
        by = u * u * sy + 2 * u * t * cy + t * t * ey2
        pts.append((int(bx), int(by)))

    if glow:
        for i in range(1, len(pts)):
            cv2.line(img, pts[i - 1], pts[i], (0, 0, 0), thickness + 3, cv2.LINE_AA)
    for i in range(1, len(pts)):
        cv2.line(img, pts[i - 1], pts[i], color, thickness, cv2.LINE_AA)
    # Arrowhead at the trimmed tip, oriented along the last bezier segment
    if len(pts) >= 2:
        cv2.arrowedLine(
            img, pts[-2], pts[-1], color, thickness,
            tipLength=0.6, line_type=cv2.LINE_AA,
        )


def draw_dashed_path(
    img: np.ndarray,
    points_px: List[Tuple[int, int]],
    color: Tuple[int, int, int],
    *,
    thickness: int = 2,
    seg_len: int = 6,
    gap: int = 4,
    fade: bool = True,
) -> None:
    """Dashed polyline through given pixel points. `fade=True` ramps alpha
    from 0.3 at the tail to 1.0 at the head, used for carry/run paths."""
    if len(points_px) < 2:
        return
    n = len(points_px)
    for i in range(1, n):
        a = points_px[i - 1]
        b = points_px[i]
        if not (a and b):
            continue
        alpha = (0.3 + 0.7 * (i / max(1, n - 1))) if fade else 1.0
        # walk segments: seg_len drawn, gap empty
        ax, ay = a
        bx, by = b
        dx, dy = bx - ax, by - ay
        length = max(1.0, float(np.hypot(dx, dy)))
        sx, sy = dx / length, dy / length
        t = 0.0
        while t < length:
            t1 = min(t + seg_len, length)
            p0 = (int(ax + sx * t), int(ay + sy * t))
            p1 = (int(ax + sx * t1), int(ay + sy * t1))
            if alpha < 1.0:
                overlay = img.copy()
                cv2.line(overlay, p0, p1, color, thickness, cv2.LINE_AA)
                cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)
            else:
                cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)
            t = t1 + gap


# ── zone hulls + banners ──────────────────────────────────────────────────

def draw_glow_line(
    img: np.ndarray,
    p_from: Tuple[int, int],
    p_to: Tuple[int, int],
    color: Tuple[int, int, int],
    *,
    thickness: int = 10,
    glow_layers: int = 3,
) -> None:
    """Thick semi-transparent line with a halo. Used to mark the touchline as
    a tactical 'extra defender' so the trap reads spatially."""
    if p_from is None or p_to is None:
        return
    # Outer halo (very thick, very transparent)
    overlay = img.copy()
    for i in range(glow_layers, 0, -1):
        t = thickness + i * 6
        a = 0.10 + 0.06 * (glow_layers - i + 1)
        cv2.line(overlay, p_from, p_to, color, t, cv2.LINE_AA)
        cv2.addWeighted(overlay, a, img, 1.0 - a, 0, img)
    # Bright core
    cv2.line(img, p_from, p_to, color, thickness, cv2.LINE_AA)


def draw_blocked_lane(
    img: np.ndarray,
    p_from: Tuple[int, int],
    p_to: Tuple[int, int],
    color: Tuple[int, int, int] = (56, 56, 240),
    *,
    thickness: int = 3,
) -> None:
    """Dashed line with a small 'X' at midpoint — 'this passing lane is closed'."""
    if p_from is None or p_to is None:
        return
    # Dashed using existing helper
    pts = [p_from, p_to]
    draw_dashed_path(img, pts, color, thickness=thickness, seg_len=10, gap=8, fade=False)
    # X marker at midpoint
    mx = (p_from[0] + p_to[0]) // 2
    my = (p_from[1] + p_to[1]) // 2
    s = 9
    cv2.line(img, (mx - s, my - s), (mx + s, my + s), (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.line(img, (mx - s, my + s), (mx + s, my - s), (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.line(img, (mx - s, my - s), (mx + s, my + s), color, thickness, cv2.LINE_AA)
    cv2.line(img, (mx - s, my + s), (mx + s, my - s), color, thickness, cv2.LINE_AA)


def draw_role_pill(
    img: np.ndarray,
    anchor_px: Tuple[int, int],
    text: str,
    color: Tuple[int, int, int],
    *,
    above_offset_px: int = 22,
    below: bool = False,
    fg: Tuple[int, int, int] = (255, 255, 255),
) -> None:
    """Clean single-piece pill above anchor (or below when `below=True`).
    No number tab, no fallback '?'. Centred horizontally on anchor."""
    if not text:
        return
    text = text.upper()
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.48
    th = 1
    (tw, hh), _ = cv2.getTextSize(text, font, fs, th)
    pad_x, pad_y = 8, 5
    box_w = tw + 2 * pad_x
    box_h = hh + 2 * pad_y
    cx, cy = anchor_px
    x1 = cx - box_w // 2
    if below:
        y1 = cy + above_offset_px
        y2 = y1 + box_h
    else:
        y2 = max(cy - above_offset_px, box_h)
        y1 = y2 - box_h

    # Black halo for legibility on grass
    cv2.rectangle(img, (x1 - 1, y1 - 1), (x1 + box_w + 1, y2 + 1), (0, 0, 0), -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1), (x1 + box_w, y2), color, -1, cv2.LINE_AA)
    cv2.putText(img, text, (x1 + pad_x, y2 - pad_y), font, fs, fg, th + 1, cv2.LINE_AA)


def draw_local_callout(
    img: np.ndarray,
    anchor_px: Tuple[int, int],
    text: str,
    *,
    side: str = "right",
    bg: Tuple[int, int, int] = (28, 28, 28),
    fg: Tuple[int, int, int] = (240, 240, 240),
    accent: Optional[Tuple[int, int, int]] = None,
) -> None:
    """Short tactical caption near a player/ring (e.g. 'PASSING LANE CLOSED').
    `side` controls whether the chip sits to the right or left of anchor."""
    if not text:
        return
    text = text.upper()
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.5
    th = 1
    (tw, hh), _ = cv2.getTextSize(text, font, fs, th)
    pad_x, pad_y = 10, 6
    box_w = tw + 2 * pad_x + (6 if accent is not None else 0)
    box_h = hh + 2 * pad_y
    cx, cy = anchor_px
    if side == "left":
        x1 = max(8, cx - 28 - box_w)
    else:
        x1 = cx + 28
    y1 = max(8, cy - box_h // 2)
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x1 + box_w, y1 + box_h), bg, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
    text_x = x1 + pad_x
    if accent is not None:
        cv2.rectangle(img, (x1, y1), (x1 + 5, y1 + box_h), accent, -1, cv2.LINE_AA)
        text_x = x1 + 5 + pad_x
    cv2.putText(img, text, (text_x, y1 + box_h - pad_y), font, fs, fg, th, cv2.LINE_AA)


def draw_zone_hull(
    img: np.ndarray,
    points_px: List[Tuple[int, int]],
    color_bgr: Tuple[int, int, int],
    *,
    alpha: float = 0.32,
    outline_thickness: int = 3,
) -> None:
    """Filled convex hull + saturated outline + thin black shadow ring.
    Used for shaded tactical zones (pressing trap, overload, defensive line, etc.).
    The shadow ring keeps the polygon legible on bright grass."""
    pts = [p for p in points_px if p is not None]
    if len(pts) < 3:
        return
    arr = np.array(pts, dtype=np.int32)
    hull = cv2.convexHull(arr)
    overlay = img.copy()
    cv2.fillPoly(overlay, [hull], color_bgr, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)
    # Black shadow underneath the saturated outline so the polygon reads on grass
    cv2.polylines(img, [hull], isClosed=True, color=(0, 0, 0),
                  thickness=outline_thickness + 2, lineType=cv2.LINE_AA)
    outline = tuple(min(255, int(c * 1.2)) for c in color_bgr)
    cv2.polylines(img, [hull], isClosed=True, color=outline,
                  thickness=outline_thickness, lineType=cv2.LINE_AA)


def draw_banner(
    img: np.ndarray,
    text: str,
    *,
    position: str = "top",
    accent: Optional[Tuple[int, int, int]] = None,
) -> None:
    """Top-centre title banner. Larger than draw_caption.
    position='top' -> big bold title at y=40; 'top_sub' -> smaller subtitle at y=78."""
    if not text:
        return
    h_img, w_img = img.shape[:2]
    is_top = position == "top"
    fs = 1.1 if is_top else 0.62
    th = 3 if is_top else 2
    y_baseline = 56 if is_top else 96
    fg = (245, 245, 245) if is_top else (210, 210, 210)
    bg = (20, 20, 20)
    font = cv2.FONT_HERSHEY_SIMPLEX

    (tw, hh), _ = cv2.getTextSize(text.upper(), font, fs, th)
    pad_x = 22 if is_top else 14
    pad_y = 12 if is_top else 8
    box_w = tw + 2 * pad_x
    box_h = hh + 2 * pad_y
    x = (w_img - box_w) // 2
    y = y_baseline - hh - pad_y

    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), bg, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.82, img, 0.18, 0, img)

    if accent is not None:
        cv2.rectangle(img, (x, y), (x + 8, y + box_h), accent, -1, cv2.LINE_AA)
        text_x = x + 8 + pad_x
    else:
        text_x = x + pad_x
    cv2.putText(img, text.upper(), (text_x, y_baseline), font, fs, fg, th, cv2.LINE_AA)


# ── caption chip ──────────────────────────────────────────────────────────

def draw_caption(
    img: np.ndarray,
    text: str,
    *,
    pos: Tuple[int, int] = (24, 24),
    bg: Tuple[int, int, int] = (20, 20, 20),
    fg: Tuple[int, int, int] = (245, 245, 245),
    accent: Optional[Tuple[int, int, int]] = None,
) -> None:
    """Top-left rounded chip with the dominant tactical label, e.g. 'BUILD UP'.
    Mirrors the X reference clip aesthetic."""
    if not text:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.7
    th = 2
    (tw, hh), _ = cv2.getTextSize(text, font, fs, th)
    pad_x, pad_y = 14, 10
    x, y = pos
    box_w = tw + 2 * pad_x
    box_h = hh + 2 * pad_y

    # shadow
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), bg, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)

    if accent is not None:
        cv2.rectangle(img, (x, y), (x + 6, y + box_h), accent, -1, cv2.LINE_AA)
        text_x = x + 6 + pad_x
    else:
        text_x = x + pad_x
    cv2.putText(img, text, (text_x, y + box_h - pad_y), font, fs, fg, th, cv2.LINE_AA)


# ── caption picker ────────────────────────────────────────────────────────

CAPTION_FOR_EVENT = {
    "shot": "CHANCE CREATED",
    "carry": "BUILD UP",
    "pass": "BUILD UP",
    "turnover": "TRANSITION",
    "tackle": "HIGH PRESS",
    "pressure": "HIGH PRESS",
    "dribble": "BUILD UP",
}


def caption_for_window(events: List[dict], frame_lo: int, frame_hi: int) -> Optional[str]:
    """Pick the dominant tactical label for a clip window.
    Priority: CHANCE CREATED > TRANSITION > HIGH PRESS > BUILD UP."""
    if not events:
        return None
    counts = {}
    for e in events:
        f = e.get("frame", e.get("frameIndex"))
        if f is None or not (frame_lo <= int(f) <= frame_hi):
            continue
        label = CAPTION_FOR_EVENT.get(str(e.get("type")), None)
        if label:
            counts[label] = counts.get(label, 0) + 1
    if not counts:
        return None
    priority = ["CHANCE CREATED", "TRANSITION", "HIGH PRESS", "BUILD UP"]
    for p in priority:
        if p in counts and counts[p] >= 1:
            return p
    return max(counts, key=counts.get)
