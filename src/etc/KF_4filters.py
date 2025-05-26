import numpy as np
from scipy.optimize import linear_sum_assignment
from tracker import MultiObjectTracker


"""
Four 6D Kalman Filter variants using only position measurements (px, py) to estimate
state [px, py, vx, vy, ax, ay].
"""

class KalmanFilterCV6:
    """Constant Velocity: ax=ay=0"""
    def __init__(self, dt=1.0, q_var=1.0, r_var=1.0):
        self.dt = dt
        # State transition matrix F for CV6
        self.F = np.array([
            [1, 0, dt, 0, 0, 0],
            [0, 1, 0, dt, 0, 0],
            [0, 0, 1,  0, 0, 0],
            [0, 0, 0,  1, 0, 0],
            [0, 0, 0,  0, 1, 0],
            [0, 0, 0,  0, 0, 1]
        ])
        # Measurement matrix: measure px, py
        self.H = np.zeros((2,6))
        self.H[0,0] = 1
        self.H[1,1] = 1
        # Covariances
        self.Q = q_var * np.eye(6)
        self.R = r_var * np.eye(2)
        # Initial state and covariance
        self.x = np.zeros((6,1))
        self.P = np.eye(6)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z):
        z = np.array(z).reshape(2,1)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        return self.x.copy()

class KalmanFilterCA6:
    """Constant Acceleration: ax, ay included"""
    def __init__(self, dt=1.0, q_var=1.0, r_var=1.0):
        self.dt = dt
        dt2 = 0.5 * dt * dt
        # State transition matrix F for CA6
        self.F = np.array([
            [1, 0, dt, 0, dt2, 0],
            [0, 1, 0, dt, 0, dt2],
            [0, 0, 1,  0, dt, 0],
            [0, 0, 0,  1, 0, dt],
            [0, 0, 0,  0, 1, 0],
            [0, 0, 0,  0, 0, 1]
        ])
        self.H = np.zeros((2,6))
        self.H[0,0] = 1
        self.H[1,1] = 1
        self.Q = q_var * np.eye(6)
        self.R = r_var * np.eye(2)
        self.x = np.zeros((6,1))
        self.P = np.eye(6)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z):
        z = np.array(z).reshape(2,1)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        return self.x.copy()

class ExtendedKalmanFilterCTRV6:
    """Constant Turn Rate & Velocity in Cartesian accel form"""
    def __init__(self, dt=1.0, q_var=1.0, r_var=1.0):
        self.dt = dt
        # measurement matrix
        self.H = np.zeros((2,6)); self.H[0,0]=1; self.H[1,1]=1
        self.Q = q_var * np.eye(6)
        self.R = r_var * np.eye(2)
        # state: [px,py,vx,vy,ax,ay]
        self.x = np.zeros((6,1))
        self.P = np.eye(6)

    def predict(self):
        # px,py updated by vx,vy
        self.x[0,0] += self.x[2,0] * self.dt
        self.x[1,0] += self.x[3,0] * self.dt
        # vx,vy updated by ax,ay
        self.x[2,0] += self.x[4,0] * self.dt
        self.x[3,0] += self.x[5,0] * self.dt
        # ax,ay remain
        F = np.eye(6)
        # fill linearized jacobian manually
        F[0,2] = self.dt; F[1,3] = self.dt
        F[2,4] = self.dt; F[3,5] = self.dt
        self.P = F @ self.P @ F.T + self.Q
        return self.x.copy()

    def update(self, z):
        z = np.array(z).reshape(2,1)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        return self.x.copy()

class ExtendedKalmanFilterCTRA6:
    """Constant Turn Rate & Acceleration in Cartesian accel form"""
    def __init__(self, dt=1.0, q_var=1.0, r_var=1.0):
        self.dt = dt
        self.H = np.zeros((2,6)); self.H[0,0]=1; self.H[1,1]=1
        self.Q = q_var * np.eye(6)
        self.R = r_var * np.eye(2)
        self.x = np.zeros((6,1))
        self.P = np.eye(6)

    def predict(self):
        # px,py
        self.x[0,0] += self.x[2,0] * self.dt
        self.x[1,0] += self.x[3,0] * self.dt
        # vx,vy updated by ax,ay
        self.x[2,0] += self.x[4,0] * self.dt
        self.x[3,0] += self.x[5,0] * self.dt
        # ax,ay remain (constant)
        F = np.eye(6)
        F[0,2] = self.dt; F[1,3] = self.dt
        F[2,4] = self.dt; F[3,5] = self.dt
        self.P = F @ self.P @ F.T + self.Q
        return self.x.copy()

    def update(self, z):
        z = np.array(z).reshape(2,1)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        return self.x.copy()

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
        else:
            raise ValueError(f"Unknown filter_type: {filter_type}")
        self.kf.x[0,0], self.kf.x[1,0] = detection

    def predict(self):
        return self.kf.predict().flatten()

    def update(self, detection):
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
        # 1) 연관 및 갱신
        matched, un_dets, un_trks = self.associate(detections)
        for trk_idx, det_idx in matched:
            self.tracks[trk_idx].update(detections[det_idx])
        # 2) unmatched 트랙 skip 증가 및 제거
        for idx in un_trks:
            self.tracks[idx].skipped_frames += 1
        self.tracks = [t for t in self.tracks if t.skipped_frames <= self.max_skipped]

        # === UPDATED LOGIC START ===
        # CHANGED: Only initialize pending before any confirmed tracks exist
        if len(self.tracks) == 0:
            new_pending = []
            for pen in self.pending:
                if any(np.linalg.norm(np.array(d)-pen['detection']) <= self.dist_threshold
                       for d in detections):
                    pen['count'] += 1
                    if pen['count'] >= self.init_frames:
                        self.tracks.append(
                            Track(pen['detection'], self.next_id,
                                  filter_type=self.filter_type,
                                  dt=self.dt, q_var=self.q_var, r_var=self.r_var)
                        )
                        self.next_id += 1
                    else:
                        new_pending.append(pen)
            for idx in un_dets:
                new_pending.append({'detection': detections[idx], 'count': 1})
            self.pending = new_pending
        else:
            # CHANGED: Clear pending to prevent additional track creation
            self.pending = []
        # === UPDATED LOGIC END ===

    def get_tracks(self):
        out = []
        for t in self.tracks:
            px, py, vx, vy, ax, ay = t.get_state()
            out.append({
                'id': t.track_id,
                'px': float(px), 'py': float(py),
                'vx': float(vx), 'vy': float(vy),
                'ax': float(ax), 'ay': float(ay)
            })
        return out
        


# --- 사용 예제 (고정값 측정) --- #
if __name__ == '__main__':
    filter_type = 'CV6'  # 'CV6', 'CA6', 'CTRV6', 'CTRA6'
    tracker = MultiObjectTracker(
        filter_type=filter_type,
        dt=1.0, q_var=1.0, r_var=1.0,
        max_skipped=3, dist_threshold=10, init_frames=2
    )
    frames = [
        [[10.0, 10.0], [20.0, 15.0], [30.0, 20.0]],
        [[11.0, 10.5], [21.0, 15.5], [31.0, 20.5]],
        [[12.0, 11.0], [22.0, 16.0], [32.0, 21.0]],
        [[13.0, 11.5], [23.0, 16.5], [33.0, 21.5]],
        [[14.0, 12.0], [24.0, 17.0], [34.0, 22.0]],
        [[15.0, 12.5], [25.0, 17.5], [35.0, 22.5]],
        [[16.0, 13.0], [26.0, 18.0], [36.0, 23.0]],
        [[17.0, 13.5], [27.0, 18.5], [37.0, 23.5]],
        [[18.0, 14.0], [28.0, 19.0], [38.0, 24.0]],
        [[19.0, 14.5], [29.0, 19.5], [39.0, 24.5]],
    ]
    for idx, dets in enumerate(frames):
        print(f"\n--- Frame {idx} ---")
        print("Detections:", dets)
        tracker.update(dets)
        tracks = tracker.get_tracks()
        if not tracks:
            print("No confirmed tracks yet.")
        else:
            for tr in tracks:
                print(
                    f"Track {tr['id']}: px={tr['px']:.1f}, py={tr['py']:.1f}, "
                    f"vx={tr['vx']:.1f}, vy={tr['vy']:.1f}, "
                    f"ax={tr['ax']:.1f}, ay={tr['ay']:.1f}"
                )
