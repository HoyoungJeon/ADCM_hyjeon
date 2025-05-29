import numpy as np
from scipy.optimize import linear_sum_assignment
from kalman_filters import (
    IMMEstimator,
    KalmanFilterCV6,
    KalmanFilterCA6,
    VariableTurnEKF,
    FixedTurnEKF,
)
import copy

class Track:
    """Single-object track using an IMM of 6D filters."""
    def __init__(self, detection, track_id, dt=1.0, q_var=1.0, r_var=1.0):
        self.track_id = track_id
        self.skipped_frames = 0

        # Create the IMM over the four models
        models = [
            KalmanFilterCV6(dt, q_var, r_var),
            KalmanFilterCA6(dt, q_var, r_var),
            VariableTurnEKF(dt, q_var, r_var),
            FixedTurnEKF(dt, q_var, r_var),
        ]
        self.kf = IMMEstimator(models)

        # Initialize all sub-models' position to the first detection
        for m in self.kf.models:
            m.x[0, 0], m.x[1, 0] = detection

        # Keep history of raw detections for yaw initialization
        self.history = [detection]
        self.dt = dt
        self._init_ctrv = False

    def predict(self):
        """Run one-step IMM predict and return the combined state."""
        self.kf.predict()
        return self.kf.x.flatten()

    def update(self, detection):
        """Incorporate a new detection into the track (IMM predict+update)."""
        # 1) Append new measurement to history
        self.history.append(detection)

        # 2) Once, initialize yaw & yaw_rate for all CTRV/EKF submodels
        if not self._init_ctrv and len(self.history) >= 2:
            x0, y0 = self.history[-2]
            x1, y1 = self.history[-1]
            yaw = np.arctan2(y1 - y0, x1 - x0)
            for m in self.kf.models:
                if isinstance(m, (VariableTurnEKF, FixedTurnEKF)):
                    m.x[3, 0] = yaw
                    m.x[4, 0] = 0.0
            self._init_ctrv = True

        # 3) IMM predict and update
        self.kf.predict()
        self.kf.update(detection)
        self.skipped_frames = 0

    def get_state(self):
        """Return the current combined state vector [px, py, vx, vy, ax, ay]."""
        return self.kf.x.flatten()


class MultiObjectTracker:
    """
    Multi-object tracker using IMM-based 6D tracks.

    - A detection must appear in `init_frames` consecutive frames to confirm a track.
    - Tracks are removed after exceeding `max_skipped` consecutive missed detections.
    """
    def __init__(
        self,
        dt=1.0,
        q_var=1.0,
        r_var=1.0,
        max_skipped=5,
        dist_threshold=10.0,
        init_frames=3
    ):
        self.dt = dt
        self.q_var = q_var
        self.r_var = r_var
        self.max_skipped = max_skipped
        self.dist_threshold = dist_threshold
        self.init_frames = init_frames

        self.tracks = []
        self.next_id = 0
        self.pending = []  # list of {'detection': [x,y], 'count': n}

    def associate(self, detections):
        """Compute assignment between existing tracks and new detections."""
        N, M = len(self.tracks), len(detections)
        if N == 0:
            return [], list(range(M)), []

        cost = np.zeros((N, M))
        for i, trk in enumerate(self.tracks):
            pred = trk.predict()
            for j, d in enumerate(detections):
                cost[i, j] = np.linalg.norm(pred[:2] - d)

        row, col = linear_sum_assignment(cost)
        matched, un_dets, un_trks = [], [], []

        for i, j in zip(row, col):
            if cost[i, j] > self.dist_threshold:
                un_trks.append(i)
                un_dets.append(j)
            else:
                matched.append((i, j))

        un_trks += [i for i in range(N) if i not in row]
        un_dets += [j for j in range(M) if j not in col]
        return matched, un_dets, un_trks

    def update(self, detections):
        """
        1) Associate and update confirmed tracks;
        2) Age and prune stale tracks;
        3) Build pending detections to confirm new tracks.
        """
        matched, un_dets, un_trks = self.associate(detections)

        # Update matched tracks
        for trk_idx, det_idx in matched:
            self.tracks[trk_idx].update(detections[det_idx])

        # Age unmatched tracks and remove those too old
        for idx in un_trks:
            self.tracks[idx].skipped_frames += 1
        self.tracks = [
            t for t in self.tracks
            if t.skipped_frames <= self.max_skipped
        ]

        # Handle pending detections
        new_pending = []
        for pen in self.pending:
            # Find closest unmatched detection
            dists = [
                (j, np.linalg.norm(np.array(detections[j]) - pen['detection']))
                for j in un_dets
            ]
            if dists:
                j_min, min_dist = min(dists, key=lambda x: x[1])
                if min_dist <= self.dist_threshold:
                    pen['detection'] = detections[j_min]
                    pen['count'] += 1
                    if pen['count'] >= self.init_frames:
                        # Confirm a new track
                        self.tracks.append(
                            Track(pen['detection'], self.next_id,
                                  dt=self.dt, q_var=self.q_var, r_var=self.r_var)
                        )
                        self.next_id += 1
                    else:
                        new_pending.append(pen)
                    un_dets.remove(j_min)
                else:
                    # Detection moved too far: keep pending but reset count
                    new_pending.append({'detection': pen['detection'], 'count': 1})
            else:
                # No candidate this frame: keep pending with count reset
                new_pending.append({'detection': pen['detection'], 'count': 1})

        # Any remaining unmatched detections start as pending
        for idx in un_dets:
            new_pending.append({'detection': detections[idx], 'count': 1})

        self.pending = new_pending

    def get_tracks(self):
        """Return a list of dicts with each track’s current state and metadata."""
        output = []
        for t in self.tracks:
            px, py, vx, vy, ax, ay = t.get_state()
            output.append({
                'id': t.track_id,
                'px': float(px), 'py': float(py),
                'vx': float(vx), 'vy': float(vy),
                'ax': float(ax), 'ay': float(ay),
                'skipped_frames': t.skipped_frames,
            })
        return output


def predict_future_tracks(tracker, steps=5):
    """
    Predict the next `steps` frames for each confirmed track using
    the single IMM sub-model with the highest probability.
    Returns a dict mapping track_id to a list of (px, py) tuples.
    """
    predictions = {}
    for trk in tracker.tracks:
        # Select the IMM sub-model with the highest weight
        best_idx = int(np.argmax(trk.kf.mu))
        model = copy.deepcopy(trk.kf.models[best_idx])

        traj = []
        for _ in range(steps):
            state = model.predict()
            px, py = float(state[0, 0]), float(state[1, 0])
            traj.append((px, py))

        predictions[trk.track_id] = traj

    return predictions
