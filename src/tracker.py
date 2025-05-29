import numpy as np
from scipy.optimize import linear_sum_assignment
from kalman_filters import *
import copy
import math

class Track:
    """Single-object track using one of the 6D Kalman filters."""
    def __init__(self, detection, track_id,
                 filter_type='CV6', dt=1.0, q_var=1.0, r_var=1.0):
        self.track_id = track_id
        self.skipped_frames = 0
        if filter_type == 'CV6':
            self.kf = KalmanFilterCV6(dt, q_var, r_var)
        elif filter_type == 'CA6':
            self.kf = KalmanFilterCA6(dt, q_var, r_var)
        elif filter_type == 'VariableTurnEKF':
            self.kf = VariableTurnEKF(dt, q_var, r_var)
        elif filter_type == 'FixedTurnEKF':
            self.kf = FixedTurnEKF(dt, q_var, r_var)
        elif filter_type == 'IMM':
            # IMM: 4가지 모델(CV6, CA6, VariableTurnEKF, VariableTurnEKF, FixedTurnEKF) 사용
            from kalman_filters import IMMEstimator
            models = [
                KalmanFilterCV6(dt, q_var, r_var),
                KalmanFilterCA6(dt, q_var, r_var),
                VariableTurnEKF(dt, q_var, r_var),
                FixedTurnEKF(dt, q_var, r_var)
            ]


            self.kf = IMMEstimator(models)
        else:
            raise ValueError(f"Unknown filter_type: {filter_type}")
        self.kf.x[0,0], self.kf.x[1,0] = detection
        if filter_type == 'IMM':
            for m in self.kf.models:
                m.x[0,0], m.x[1,0] = detection

        self.history = [detection]
        self.dt = dt
        self._init_ctrv = False

    def predict(self):
        self.kf.predict()
        return self.kf.x.flatten()

    def update(self, detection):
        # self.kf.update(detection)
        #
        #
        # self.kf.predict()
        # self.skipped_frames = 0


        # 1) 히스토리 추가
        self.history.append(detection)


        # 2) CTRV용 yaw/yaw_rate 단 한 번만 초기화
        if (not self._init_ctrv
            and isinstance(self.kf, (VariableTurnEKF, FixedTurnEKF))
            and len(self.history) >= 2):
            # 첫 두 점으로 yaw 초기화
            x0,y0 = self.history[-2]
            x1,y1 = self.history[-1]
            yaw = np.arctan2(y1-y0, x1-x0)
            self.kf.x[3,0] = yaw
            self.kf.x[4,0] = 0.0
            self._init_ctrv = True

        # 3) 예측·업데이트
        self.kf.predict()
        self.kf.update(detection)
        self.skipped_frames = 0

    def get_state(self):
        return self.kf.x.flatten()

class MultiObjectTracker:
    """
    Multi-object tracker with 6D KF-based Track.
    - init_frames 번 연속 검출되어야 트랙으로 확정.
    - max_skipped 초과하면 트랙 제거.
    """
    def __init__(self,
                 filter_type='CV6', dt=1.0, q_var=1.0, r_var=1.0,
                 max_skipped=5, dist_threshold=10.0, init_frames=3):
        self.filter_type = filter_type
        self.dt, self.q_var, self.r_var = dt, q_var, r_var
        self.max_skipped = max_skipped
        self.dist_threshold = dist_threshold
        self.init_frames = init_frames

        self.tracks = []
        self.next_id = 0
        self.pending = []  # [{'detection': [x,y], 'count': n}, ...]

    def associate(self, detections):
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
                un_trks.append(i); un_dets.append(j)
            else:
                matched.append((i, j))
        un_trks += [i for i in range(N) if i not in row]
        un_dets += [j for j in range(M) if j not in col]
        return matched, un_dets, un_trks

    def update(self, detections):
        # 1) Associate and update confirmed tracks
        matched, un_dets, un_trks = self.associate(detections)
        for trk_idx, det_idx in matched:
            self.tracks[trk_idx].update(detections[det_idx])

        # 2) Increment skip for unmatched confirmed tracks and remove stale
        for idx in un_trks:
            self.tracks[idx].skipped_frames += 1
        self.tracks = [t for t in self.tracks if t.skipped_frames <= self.max_skipped]

        # 3) Pending detection logic (always active)
        new_pending = []
        for pen in self.pending:
            # 아직 매칭되지 않은 검출들에 대해 (인덱스, 거리) 쌍을 계산
            dists = [
                (j, np.linalg.norm(np.array(detections[j]) - pen['detection']))
                for j in un_dets
            ]
            # 거리 최소값을 찾고, 그 값이 문턱 이하일 때만 매칭
            if dists:
                j_min, min_dist = min(dists, key=lambda x: x[1])
                if min_dist <= self.dist_threshold:
                    matched_det_idx = j_min
                else:
                    matched_det_idx = None
            else:
                matched_det_idx = None

            if matched_det_idx is not None:
                pen['detection'] = detections[matched_det_idx]
                pen['count'] += 1
                if pen['count'] >= self.init_frames:
                    # Confirm and create new track
                    self.tracks.append(
                        Track(
                            pen['detection'], self.next_id,
                            filter_type=self.filter_type,
                            dt=self.dt, q_var=self.q_var, r_var=self.r_var
                        )
                    )
                    self.next_id += 1

                else:
                    # still need more consecutive frames, keep in pending
                    new_pending.append(pen)
                if matched_det_idx in un_dets:
                    un_dets.remove(matched_det_idx)
            else:
                # no match this frame, reset or keep based on policy
                new_pending.append(pen)

        # Add any remaining unmatched detections as new pending
        for idx in un_dets:
            new_pending.append({'detection': detections[idx], 'count': 1})
        self.pending = new_pending

    def get_tracks(self):
        out = []
        for t in self.tracks:
            px, py, vx, vy, ax, ay = t.get_state()
            out.append({
                'id': t.track_id,
                'px': float(px), 'py': float(py),
                'vx': float(vx), 'vy': float(vy),
                'ax': float(ax), 'ay': float(ay),
                'skipped_frames': t.skipped_frames,
            })
        return out

def predict_future_tracks(tracker, steps=5):
    # Best model selection
    """
    For each track, pick the single IMM sub‐model with highest μ (or the sole filter if non‐IMM),
    deep‐copy it, and run N‐step predictions.
    Returns a dict mapping track_id to a list of (px, py) predictions.
    """
    predictions = {}

    for trk in tracker.tracks:
        track_id = trk.track_id
        kf = trk.kf

        # 1) If it's an IMM, select the sub‐model with highest μ
        if isinstance(kf, IMMEstimator):
            best_idx = int(np.argmax(kf.mu))
            model = copy.deepcopy(kf.models[best_idx])
        else:
            # 2) otherwise just deep‐copy the single filter/ekf
            model = copy.deepcopy(kf)

        # 3) Run N‐step predict on that model
        traj = []
        for _ in range(steps):
            state = model.predict()
            px, py = float(state[0, 0]), float(state[1, 0])
            traj.append((px, py))

        predictions[track_id] = traj

    return predictions

