from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings.
    
    Returns: float in range [-1, 1], where 1 = identical, 0 = orthogonal, -1 = opposite
    """
    if emb1.size == 0 or emb2.size == 0:
        return 0.0
    
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    
    return float(np.dot(emb1, emb2) / (norm1 * norm2))


@dataclass
class DriftTracker:
    """Tracks embedding similarity drift per player ID."""
    
    drift_threshold: float = 0.70
    pid_anchors: Dict[str, np.ndarray] = field(default_factory=dict)
    pid_history: Dict[str, List[float]] = field(default_factory=dict)
    pid_decision_log: Dict[str, List[Dict]] = field(default_factory=dict)
    
    def create_anchor(self, pid: str, embedding: Optional[np.ndarray]) -> None:
        """Store initial embedding for a player."""
        if embedding is not None and embedding.size > 0:
            self.pid_anchors[pid] = embedding.copy()
            self.pid_history[pid] = []
            self.pid_decision_log[pid] = []
    
    def update_drift(self, pid: str, embedding: Optional[np.ndarray]) -> Optional[float]:
        """Compute and record drift for a player. Returns similarity score or None."""
        if pid not in self.pid_anchors or embedding is None or embedding.size == 0:
            return None
        
        anchor = self.pid_anchors[pid]
        similarity = compute_cosine_similarity(anchor, embedding)
        self.pid_history[pid].append(similarity)
        return similarity
    
    def should_trigger_vlm(self, pid: str, similarity: Optional[float]) -> bool:
        """Decide if VLM should be called based on drift."""
        if similarity is None:
            return False
        return similarity < self.drift_threshold
    
    def log_vlm_decision(self, pid: str, frame: int, similarity: float, triggered: bool) -> None:
        """Log VLM decision for audit."""
        self.pid_decision_log[pid].append({
            "frame": frame,
            "similarity": similarity,
            "triggered": triggered,
        })
    
    def export_report(self) -> Dict:
        """Export drift tracking data as JSON-serializable dict."""
        report = {
            "config": {
                "drift_threshold": self.drift_threshold,
            },
            "players": {},
        }
        
        for pid in self.pid_anchors:
            history = self.pid_history.get(pid, [])
            decisions = self.pid_decision_log.get(pid, [])
            
            report["players"][pid] = {
                "similarity_history": history,
                "decision_log": decisions,
                "final_stability": history[-1] if history else None,
                "total_frames": len(history),
                "drift_triggered_count": sum(1 for d in decisions if d.get("triggered")),
            }
        
        return report
