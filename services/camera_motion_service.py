"""
Camera motion compensation service.

Uses sparse optical flow on stable pitch-edge features to estimate
frame-to-frame affine transforms, then compensates track predictions.
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)


class CameraMotionCompensator:

    def __init__(self):
        self.prev_frame_gray = None

    def compute_compensation(self, frame_gray: np.ndarray) -> np.ndarray:
        """
        Compute affine transform between current and previous frame
        using sparse optical flow on stable pitch-edge features.
        Returns 2x3 affine matrix. Returns identity on failure.
        """
        identity = np.eye(2, 3, dtype=np.float32)

        try:
            import cv2
        except ImportError:
            self.prev_frame_gray = frame_gray
            return identity

        try:
            if self.prev_frame_gray is None:
                self.prev_frame_gray = frame_gray
                return identity

            if self.prev_frame_gray.shape != frame_gray.shape:
                self.prev_frame_gray = frame_gray
                return identity

            h, w = frame_gray.shape[:2]

            # Mask that zeros out the centre 50% of the frame
            # (players move there — only use outer pitch markings)
            mask = np.zeros((h, w), dtype=np.uint8)
            h_quarter = h // 4
            w_quarter = w // 4
            # Top strip
            mask[:h_quarter, :] = 255
            # Bottom strip
            mask[h - h_quarter:, :] = 255
            # Left strip
            mask[:, :w_quarter] = 255
            # Right strip
            mask[:, w - w_quarter:] = 255

            # Detect corners in previous frame
            corners = cv2.goodFeaturesToTrack(
                self.prev_frame_gray,
                maxCorners=200,
                qualityLevel=0.01,
                minDistance=1,
                mask=mask,
            )

            if corners is None or len(corners) < 4:
                self.prev_frame_gray = frame_gray
                return identity

            # Track with Lucas-Kanade optical flow
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_frame_gray,
                frame_gray,
                corners,
                None,
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
            )

            if next_pts is None or status is None:
                self.prev_frame_gray = frame_gray
                return identity

            # Forward-backward error check
            back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
                frame_gray,
                self.prev_frame_gray,
                next_pts,
                None,
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
            )

            if back_pts is None or back_status is None:
                self.prev_frame_gray = frame_gray
                return identity

            # Keep only inliers where forward-backward error < 0.5px
            good_mask = status.flatten() == 1
            if back_status is not None:
                good_mask &= back_status.flatten() == 1

            fb_error = np.sqrt(
                np.sum((corners[good_mask] - back_pts[good_mask]) ** 2, axis=-1)
            ).flatten()
            inlier_mask = fb_error < 0.5

            src_pts = corners[good_mask][inlier_mask]
            dst_pts = next_pts[good_mask][inlier_mask]

            if len(src_pts) < 4:
                self.prev_frame_gray = frame_gray
                return identity

            # Estimate affine with RANSAC
            M, inliers = cv2.estimateAffinePartial2D(
                src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0
            )

            self.prev_frame_gray = frame_gray

            if M is not None:
                return M.astype(np.float32)
            return identity

        except Exception as e:
            logger.debug(f"Camera motion compensation failed: {e}")
            self.prev_frame_gray = frame_gray
            return identity

    def compensate_tracks(self, tracks: list, affine: np.ndarray) -> list:
        """
        Apply inverse of camera motion to each track's predicted
        bounding box centre. Only compensate tracks with
        confirmed_detections > 3. Return modified tracks.
        """
        try:
            # Check if affine is identity (no compensation needed)
            identity = np.eye(2, 3, dtype=np.float32)
            if np.allclose(affine, identity, atol=1e-4):
                return tracks

            # Compute inverse affine
            # For a 2x3 partial affine [a b tx; c d ty], invert:
            a, b, tx = affine[0]
            c, d, ty = affine[1]
            det = a * d - b * c
            if abs(det) < 1e-6:
                return tracks

            inv_a = d / det
            inv_b = -b / det
            inv_c = -c / det
            inv_d = a / det
            inv_tx = -(inv_a * tx + inv_b * ty)
            inv_ty = -(inv_c * tx + inv_d * ty)

            for track in tracks:
                confirmed = track.get("confirmed_detections", track.get("_confirmed_detections", 0))
                if confirmed <= 3:
                    continue

                traj = track.get("trajectory", [])
                if not traj:
                    continue

                last_entry = traj[-1]
                bbox = last_entry.get("bbox")
                if bbox is None:
                    continue

                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0

                # Apply inverse affine to centre
                new_cx = inv_a * cx + inv_b * cy + inv_tx
                new_cy = inv_c * cx + inv_d * cy + inv_ty

                # Shift bbox by the difference
                dx = new_cx - cx
                dy = new_cy - cy
                track["_cam_compensated_bbox"] = [
                    bbox[0] + dx, bbox[1] + dy,
                    bbox[2] + dx, bbox[3] + dy,
                ]

        except Exception as e:
            logger.debug(f"Track compensation failed: {e}")

        return tracks
