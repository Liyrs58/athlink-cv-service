import cv2
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
import time


class Player:
    def __init__(self, pid):
        self.id = pid
        self.embedding = None
        self.embedding_history = deque(maxlen=20)
        self.last_pos = None
        self.last_seen = -1
        self.team = None
        self.confidence = 0.0

    def update(self, embedding, pos, frame_id, conf=1.0):
        self.embedding = embedding / (np.linalg.norm(embedding) + 1e-6)
        self.embedding_history.append(self.embedding.copy())
        self.last_pos = pos
        self.last_seen = frame_id
        self.confidence = conf

    def get_mean_embedding(self):
        if len(self.embedding_history) == 0:
            return None
        embs = np.array(list(self.embedding_history))
        mean = embs.mean(axis=0)
        return mean / (np.linalg.norm(mean) + 1e-6)


class ProductionTracker:
    def __init__(self, device="cuda"):
        self.device = device
        print("[Tracker] Loading YOLOv8n...")
        self.yolo = YOLO("yolov8n.pt")
        self.yolo.to(device)

        print("[Tracker] Loading ReID model...")
        self.reid_model = self._load_reid_model()
        self.reid_model.to(device).eval()

        self.players = [Player(i) for i in range(22)]
        self.lost_buffer = deque(maxlen=40)
        self.frame_count = 0
        self.last_reid_frame = -3
        self.last_embeddings = None
        self.last_dets = []
        self.track_to_player = {}
        self.bootstrap_frames = 0

    def _load_reid_model(self):
        try:
            from torchvision.models import resnet50
            model = resnet50(pretrained=True)
            model.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
            model.fc = torch.nn.Identity()
            return model
        except:
            print("[Tracker] Failed to load ReID, using dummy")
            class DummyReID(torch.nn.Module):
                def forward(self, x):
                    B = x.shape[0]
                    return torch.randn(B, 2048, device=x.device)
            return DummyReID()

    def detect(self, frame):
        results = self.yolo(frame, conf=0.35, verbose=False)
        dets = []
        for box in results[0].boxes:
            cls = int(box.cls.item())
            if cls in [0, 1, 2]:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf.item())
                dets.append({
                    'bbox': np.array([x1, y1, x2, y2], dtype=np.float32),
                    'conf': conf,
                    'cls': cls
                })
        return dets[:22]

    def crop_and_resize(self, frame, dets):
        crops = []
        valid_dets = []
        for det in dets:
            x1, y1, x2, y2 = det['bbox'].astype(np.int32)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            if x2 > x1 + 10 and y2 > y1 + 10:
                crop = frame[y1:y2, x1:x2]
                crop = cv2.resize(crop, (192, 96))
                crop = crop.astype(np.float32) / 255.0
                crop = np.transpose(crop, (2, 0, 1))
                crops.append(crop)
                valid_dets.append(det)

        return np.array(crops, dtype=np.float32) if crops else None, valid_dets

    def compute_embeddings(self, crops):
        if crops is None or len(crops) == 0:
            return None

        batch = torch.from_numpy(crops).to(self.device)
        if batch.shape[1] == 3:
            batch = batch * 255.0

        with torch.no_grad():
            embeddings = self.reid_model(batch)
            embeddings = F.normalize(embeddings, p=2, dim=1)

        embeddings = embeddings.cpu().numpy().astype(np.float32)
        for i in range(len(embeddings)):
            embeddings[i] = embeddings[i] / (np.linalg.norm(embeddings[i]) + 1e-6)

        return embeddings

    def compute_similarity_matrix(self, embeddings, players):
        if embeddings is None or len(embeddings) == 0:
            return np.zeros((0, 22), dtype=np.float32)

        player_embeds = []
        for p in players:
            emb = p.get_mean_embedding()
            if emb is None:
                emb = np.zeros(embeddings.shape[1], dtype=np.float32)
            player_embeds.append(emb)

        player_embeds = np.array(player_embeds, dtype=np.float32)
        similarity = (embeddings @ player_embeds.T).astype(np.float32)
        return np.clip(similarity, -1.0, 1.0)

    def hungarian_match(self, similarity):
        cost = 1.0 - similarity
        cost = np.clip(cost, 0, 2)
        row, col = linear_sum_assignment(cost)
        return list(zip(row, col))

    def uncertainty_lock(self, similarity, assignments, frame_id):
        valid = []
        for track_idx, player_idx in assignments:
            best = float(similarity[track_idx, player_idx])

            if self.players[player_idx].embedding is None:
                valid.append((track_idx, player_idx))
                continue

            all_scores = similarity[track_idx]
            scores_sorted = np.sort(all_scores)[::-1]
            second_best = float(scores_sorted[1]) if len(scores_sorted) > 1 else -1

            margin = best - second_best

            # Bootstrap phase: strict margins to lock in initial assignments
            # Normal phase: relax threshold to allow ongoing matching
            threshold = 0.02 if frame_id < 30 else 0.01

            if best < 0.3:
                continue

            if margin >= threshold:
                valid.append((track_idx, player_idx))

        return valid

    def recover_lost_player(self, embedding, used_ids):
        best_id = None
        best_sim = 0.55

        for lost_entry in self.lost_buffer:
            if lost_entry['id'] in used_ids:
                continue
            lost_emb = lost_entry['embedding']
            sim = float(np.dot(embedding, lost_emb))
            if sim > best_sim:
                best_sim = sim
                best_id = lost_entry['id']

        return best_id

    def update_players(self, assignments, embeddings, dets, frame_id):
        used_player_ids = set()
        track_to_player = {}

        for track_idx, player_idx in assignments:
            if player_idx in used_player_ids:
                continue

            emb = embeddings[track_idx]
            bbox = dets[track_idx]['bbox']
            conf = dets[track_idx]['conf']
            pos = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

            self.players[player_idx].update(emb, pos, frame_id, conf)
            used_player_ids.add(player_idx)
            track_to_player[track_idx] = player_idx

        for track_idx in range(len(embeddings)):
            if track_idx not in track_to_player:
                emb = embeddings[track_idx]
                recovered_id = self.recover_lost_player(emb, used_player_ids)

                if recovered_id is not None and recovered_id not in used_player_ids:
                    bbox = dets[track_idx]['bbox']
                    conf = dets[track_idx]['conf']
                    pos = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                    self.players[recovered_id].update(emb, pos, frame_id, conf)
                    used_player_ids.add(recovered_id)
                    track_to_player[track_idx] = recovered_id

        for i, p in enumerate(self.players):
            if i not in used_player_ids and p.last_seen >= 0:
                if frame_id - p.last_seen > 150:
                    mean_emb = p.get_mean_embedding()
                    if mean_emb is not None:
                        self.lost_buffer.append({
                            'id': i,
                            'embedding': mean_emb,
                            'last_seen': p.last_seen
                        })
                    p.embedding = None
                    p.embedding_history.clear()
                    p.last_seen = -1

        return track_to_player

    def process_frame(self, frame, frame_id):
        dets = self.detect(frame)

        if len(dets) == 0:
            return {}

        compute_reid = (frame_id - self.last_reid_frame) >= 3

        if compute_reid:
            crops, valid_dets = self.crop_and_resize(frame, dets)
            if crops is not None and len(crops) > 0:
                embeddings = self.compute_embeddings(crops)
                self.last_embeddings = embeddings
                self.last_dets = valid_dets
                self.last_reid_frame = frame_id
            else:
                embeddings = None
                valid_dets = []
        else:
            embeddings = self.last_embeddings
            valid_dets = self.last_dets

        if embeddings is None or len(embeddings) == 0:
            return {}

        similarity = self.compute_similarity_matrix(embeddings, self.players)
        assignments = self.hungarian_match(similarity)
        assignments = self.uncertainty_lock(similarity, assignments, frame_id)
        track_to_player = self.update_players(assignments, embeddings, valid_dets, frame_id)

        if frame_id % 30 == 0:
            active = sum(1 for p in self.players if p.last_seen >= 0)
            print(f"[Frame {frame_id}] Active: {active}/22 | Assignments: {len(assignments)} | Lost: {len(self.lost_buffer)}")

        self.frame_count += 1
        return track_to_player

    def draw_frame(self, frame, track_to_player, dets):
        _, valid_dets = self.crop_and_resize(frame, dets)

        for track_idx, player_idx in track_to_player.items():
            if track_idx < len(valid_dets):
                bbox = valid_dets[track_idx]['bbox'].astype(np.int32)
                x1, y1, x2, y2 = bbox

                player = self.players[player_idx]
                color = (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"P{player_idx+1}"
                cv2.putText(frame, label, (x1, max(y1-5, 0)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame


def main():
    video_path = "/content/1b16c594_villa_psg_40s_new.mp4"
    output_path = "/content/output_tracked.mp4"

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    tracker = ProductionTracker(device="cuda")

    frame_id = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        dets = tracker.detect(frame)
        track_to_player = tracker.process_frame(frame, frame_id)
        frame_drawn = tracker.draw_frame(frame, track_to_player, dets)
        out.write(frame_drawn)

        if frame_id % 10 == 0:
            elapsed = time.time() - t0
            fps_actual = (frame_id + 1) / elapsed if elapsed > 0 else 0
            print(f"Frame {frame_id:4d} | FPS: {fps_actual:5.1f} | Tracks: {len(track_to_player):2d}")

        frame_id += 1

    cap.release()
    out.release()

    elapsed = time.time() - t0
    print(f"\n✓ Done. {frame_id} frames in {elapsed:.1f}s ({frame_id/elapsed:.1f} FPS)")
    print(f"✓ Output: {output_path}")


if __name__ == "__main__":
    main()
