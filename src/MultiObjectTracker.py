import numpy as np
from scipy.optimize import linear_sum_assignment

class KalmanFilterCV:
    """
    Constant Velocity Kalman Filter for tracking [x, y, vx, vy].
    Measurements: [x, y]
    """
    def __init__(self, dt=1.0, process_var=1.0, measure_var=1.0):
        self.dt = dt
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1,  0],
                           [0, 0, 0,  1]])
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        self.Q = process_var * np.eye(4)
        self.R = measure_var * np.eye(2)
        self.x = np.zeros((4,1))
        self.P = np.eye(4)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.flatten()

    def update(self, z):
        z = np.array(z).reshape((2,1))
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

class Track:
    """Single-object confirmed track using KalmanFilterCV."""
    def __init__(self, detection, track_id, dt=1.0):
        self.kf = KalmanFilterCV(dt)
        self.kf.x[0,0], self.kf.x[1,0] = detection
        self.track_id = track_id
        self.skipped_frames = 0

    def predict(self):
        return self.kf.predict()

    def update(self, detection):
        self.kf.update(detection)
        self.skipped_frames = 0

    def get_state(self):
        return self.kf.x.flatten()

class MultiObjectTracker:
    """
    Multi-object tracker with confirmation logic:
    - Detections must persist for 'init_frames' before track creation.
    - Confirmed tracks are removed if unmatched for > max_skipped frames.
    Detection format: [x, y]
    """
    # def __init__(self, max_skipped=5, dist_threshold=50.0, init_frames=1):
    """
    Multi-object tracker with confirmation logic:
    - Detections must persist for 'init_frames' before track creation.
    - Confirmed tracks are removed if unmatched for > max_skipped frames.
    Detection format: [x, y]
    """
    def __init__(self, max_skipped=5, dist_threshold=50.0, init_frames=3):
        self.tracks = []
        self.next_id = 0
        self.max_skipped = max_skipped
        self.dist_threshold = dist_threshold
        self.init_frames = init_frames
        self.pending = []  # list of {'detection': [x,y], 'count': int}

    def associate(self, detections):
        N = len(self.tracks)
        M = len(detections)
        if N == 0:
            return [], list(range(M)), []
        cost = np.zeros((N, M))
        for i, trk in enumerate(self.tracks):
            pred = trk.predict()
            for j, det in enumerate(detections):
                cost[i, j] = np.linalg.norm(pred[:2] - det)
        row, col = linear_sum_assignment(cost)
        matched, unmatched_dets, unmatched_trks = [], [], []
        for i, j in zip(row, col):
            if cost[i, j] > self.dist_threshold:
                unmatched_trks.append(i)
                unmatched_dets.append(j)
            else:
                matched.append((i, j))
        unmatched_trks += [i for i in range(N) if i not in row]
        unmatched_dets += [j for j in range(M) if j not in col]
        return matched, unmatched_dets, unmatched_trks

    def update(self, detections):
        # Confirmed track update
        matched, unmatched_dets, unmatched_trks = self.associate(detections)
        for trk_idx, det_idx in matched:
            self.tracks[trk_idx].update(detections[det_idx])
        # Increment skip for unmatched confirmed tracks
        for idx in unmatched_trks:
            self.tracks[idx].skipped_frames += 1
        # Remove stale confirmed tracks
        self.tracks = [t for t in self.tracks if t.skipped_frames <= self.max_skipped]

                # Pending detection logic
        new_pending = []
        # Update pending counts: match by distance threshold
        for pen in self.pending:
            # find detection within distance threshold of pending detection
            matched_det = None
            for d in detections:
                if np.linalg.norm(np.array(d) - np.array(pen['detection'])) <= self.dist_threshold:
                    matched_det = d
                    break
            if matched_det is not None:
                pen['count'] += 1
                # promote to confirmed track when count reaches init_frames
                if pen['count'] >= self.init_frames:
                    self.tracks.append(Track(pen['detection'], self.next_id))
                    self.next_id += 1
                else:
                    new_pending.append(pen)
        # Add new pending for unmatched detections
        for idx in unmatched_dets:
            new_pending.append({'detection': detections[idx], 'count': 1})
        self.pending = new_pending

    def get_tracks(self):
        out = []
        for t in self.tracks:
            s = t.get_state()
            out.append({
                'id': t.track_id,
                'x': float(s[0]),
                'y': float(s[1]),
                'vx': float(s[2]),
                'vy': float(s[3])
            })
        return out

# Example usage
def main():
    tracker = MultiObjectTracker()
    frames = [
        [[10,10], [20,15], [30,20]],
        [[11,10.5], [21,15.2], [30,20.3]],
        [[12,11],   [21.5,15.4]],
        [[13,11.5], [22,15.6]],
        [[14,12]]
    ]
    for frame_idx, dets in enumerate(frames):
        print(f"Frame {frame_idx} detections: {dets}")
        tracker.update(dets)
        tracks = tracker.get_tracks()
        print("Tracked states:")
        for tr in tracks:
            print(f"  Track {tr['id']}: x={tr['x']:.2f}, y={tr['y']:.2f}")
        print()

if __name__ == '__main__':
    main()
