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
