import numpy as np
from scipy.optimize import linear_sum_assignment
from kalman_filters import *
import copy

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
        elif filter_type == 'CTRV6':
            self.kf = ExtendedKalmanFilterCTRV6(dt, q_var, r_var)
        elif filter_type == 'CTRA6':
            self.kf = ExtendedKalmanFilterCTRA6(dt, q_var, r_var)
        elif filter_type == 'IMM':
            # IMM: 4가지 모델(CV6, CA6, CTRV6, CTRA6) 사용
            from kalman_filters import IMMEstimator
            models = [
                KalmanFilterCV6(dt, q_var, r_var),
                KalmanFilterCA6(dt, q_var, r_var),
                ExtendedKalmanFilterCTRV6(dt, q_var, r_var),
                ExtendedKalmanFilterCTRA6(dt, q_var, r_var),
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
            and isinstance(self.kf, (ExtendedKalmanFilterCTRV6, ExtendedKalmanFilterCTRA6))
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
    """
    Predict the next `steps` frames for each confirmed track in the tracker.
    Returns a dict mapping track_id to a list of (px, py) predictions.
    """
    predictions = {}
    for trk in tracker.tracks:  # Track 객체 리스트 :contentReference[oaicite:0]{index=0}
        track_id = trk.track_id
        # IMMEstimator 인스턴스인지 확인 :contentReference[oaicite:1]{index=1}
        if hasattr(trk.kf, 'models'):
            # 내부 모델들과 확률 μ 복제
            models = [copy.deepcopy(m) for m in trk.kf.models]
            mu = trk.kf.mu.copy()
            traj = []
            for _ in range(steps):
                # 각 모델 predict 호출
                for m in models:
                    m.predict()
                # 모델별 상태 weighted sum
                x_comb = sum(mu[i] * models[i].x for i in range(len(models)))
                traj.append((float(x_comb[0]), float(x_comb[1])))
            predictions[track_id] = traj
        else:
            # 단일 칼만필터 복제 후 predict
            kf_copy = copy.deepcopy(trk.kf)
            traj = []
            for _ in range(steps):
                kf_copy.predict()
                x = kf_copy.x
                traj.append((float(x[0]), float(x[1])))
            predictions[track_id] = traj

    return predictions
