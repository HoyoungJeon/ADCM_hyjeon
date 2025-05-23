import numpy as np
from scipy.optimize import linear_sum_assignment

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
class IMM6:
    """4개 6D 모델(CV6, CA6, CTRV6, CTRA6)을 혼합하는 IMM 필터."""
    def __init__(self, filter_constructors, PI=None):
        self.M = len(filter_constructors)
        # 각 모델 인스턴스 생성
        self.filters = [ctor() for ctor in filter_constructors]
        # 초기 모델 확률은 균일
        self.mu = np.ones(self.M) / self.M
        # 전이 행렬: 대각 우선 확률 높게, 나머지는 균등 분배
        if PI is None:
            p = 0.90
            PI = np.full((self.M, self.M), (1-p)/(self.M-1))
            np.fill_diagonal(PI, p)
        self.PI = PI

    def predict(self):
        # Mixing 확률 μ̃_{i→j} = π_{ij} μ_i / c_j
        c_j = self.PI.T @ self.mu
        mu_ij = np.zeros((self.M, self.M))
        for i in range(self.M):
            for j in range(self.M):
                mu_ij[i,j] = self.PI[i,j] * self.mu[i] / c_j[j]
        # 섞인 상태/공분산 계산
        mixed_x, mixed_P = [], []
        for j in range(self.M):
            xj = sum(mu_ij[i,j] * self.filters[i].x for i in range(self.M))
            Pj = sum(mu_ij[i,j] * (self.filters[i].P +
                     (self.filters[i].x - xj) @ (self.filters[i].x - xj).T)
                     for i in range(self.M))
            mixed_x.append(xj)
            mixed_P.append(Pj)
        # 각 모델에 섞인 초기화 후 predict
        for j, f in enumerate(self.filters):
            f.x = mixed_x[j].copy()
            f.P = mixed_P[j].copy()
            f.predict()
        # 다음 update 단계에서 사용할 c_j 저장
        self.c_j = c_j
        return [f.x.copy() for f in self.filters]

    def update(self, z):
        # 각 모델별 likelihood 계산 및 KF 업데이트
        z = np.array(z).reshape(2,1)
        likelihood = np.zeros(self.M)
        for j, f in enumerate(self.filters):
            y = z - f.H @ f.x
            S = f.H @ f.P @ f.H.T + f.R
            exponent = float(-0.5 * y.T @ np.linalg.inv(S) @ y)
            denom = np.sqrt((2*np.pi)**2 * np.linalg.det(S))
            likelihood[j] = np.exp(exponent) / denom
            # KF update
            K = f.P @ f.H.T @ np.linalg.inv(S)
            f.x = f.x + K @ y
            f.P = (np.eye(6) - K @ f.H) @ f.P
        # 모델 확률 갱신
        mu_temp = self.c_j * likelihood
        self.mu = mu_temp / np.sum(mu_temp)
        # 결합 상태 & 공분산
        x_comb = sum(self.mu[j] * self.filters[j].x for j in range(self.M))
        P_comb = sum(self.mu[j] * (self.filters[j].P +
                 (self.filters[j].x - x_comb) @ (self.filters[j].x - x_comb).T)
                 for j in range(self.M))
        return x_comb, P_comb

# --- IMM 기반 단일 트랙 --- #
class TrackIMM6:
    def __init__(self, detection, track_id, dt=1.0, q_var=1.0, r_var=1.0):
        self.track_id = track_id
        self.skipped_frames = 0
        # 4개 6D 필터 생성자 묶음
        ctors = [
            lambda: KalmanFilterCV6(dt, q_var, r_var),
            lambda: KalmanFilterCA6(dt, q_var, r_var),
            lambda: ExtendedKalmanFilterCTRV6(dt, q_var, r_var),
            lambda: ExtendedKalmanFilterCTRA6(dt, q_var, r_var),
        ]
        self.imm = IMM6(ctors)
        # 초기 위치로 모든 모델 상태 초기화
        for f in self.imm.filters:
            f.x[0,0], f.x[1,0] = detection

    def predict(self):
        x_preds = self.imm.predict()
        # 예측된 모델별 x_preds와 mu로 결합 예측값 반환
        x_comb = sum(self.imm.mu[j] * x_preds[j] for j in range(self.imm.M))
        return x_comb.flatten()

    def update(self, detection):
        self.imm.update(detection)
        self.skipped_frames = 0

    def get_state(self):
        # 결합 추정 상태 반환
        x_comb = sum(self.imm.mu[j] * self.imm.filters[j].x
                     for j in range(self.imm.M))
        return x_comb.flatten()

# --- IMM 기반 MultiObjectTracker --- #
class MultiObjectTrackerIMM6:
    def __init__(self, dt=1.0, q_var=1.0, r_var=1.0,
                 max_skipped=5, dist_threshold=50.0, init_frames=3):
        self.dt, self.q_var, self.r_var = dt, q_var, r_var
        self.max_skipped = max_skipped
        self.dist_threshold = dist_threshold
        self.init_frames = init_frames
        self.tracks = []
        self.next_id = 0
        self.pending = []

    def associate(self, detections):
        N, M = len(self.tracks), len(detections)
        if N == 0:
            return [], list(range(M)), []
        cost = np.zeros((N, M))
        for i, trk in enumerate(self.tracks):
            pred = trk.predict()
            for j, det in enumerate(detections):
                cost[i,j] = np.linalg.norm(pred[:2] - det)
        row, col = linear_sum_assignment(cost)
        matched, un_dets, un_trks = [], [], []
        for i,j in zip(row,col):
            if cost[i,j] > self.dist_threshold:
                un_trks.append(i); un_dets.append(j)
            else:
                matched.append((i,j))
        un_trks += [i for i in range(N) if i not in row]
        un_dets += [j for j in range(M) if j not in col]
        return matched, un_dets, un_trks

    def update(self, detections):
        matched, un_dets, un_trks = self.associate(detections)
        # 1) matched 트랙 갱신
        for trk_idx, det_idx in matched:
            self.tracks[trk_idx].update(detections[det_idx])
        # 2) unmatched 트랙 skip 증가 및 제거
        for idx in un_trks:
            self.tracks[idx].skipped_frames += 1
        self.tracks = [t for t in self.tracks if t.skipped_frames <= self.max_skipped]
        # 3) pending 처리 → confirmed track 생성
        new_pending = []
        for pen in self.pending:
            if any(np.linalg.norm(np.array(d)-pen['detection']) <= self.dist_threshold
                   for d in detections):
                pen['count'] += 1
                if pen['count'] >= self.init_frames:
                    self.tracks.append(
                        TrackIMM6(pen['detection'], self.next_id,
                                  dt=self.dt, q_var=self.q_var, r_var=self.r_var)
                    )
                    self.next_id += 1
                else:
                    new_pending.append(pen)
        for idx in un_dets:
            new_pending.append({'detection': detections[idx], 'count': 1})
        self.pending = new_pending

    def get_tracks(self):
        out = []
        for t in self.tracks:
            s = t.get_state()
            out.append({
                'id': t.track_id,
                'px': float(s[0]), 'py': float(s[1]),
                'vx': float(s[2]), 'vy': float(s[3]),
                'ax': float(s[4]), 'ay': float(s[5])
            })
        return out

# --- 사용 예제 --- #
if __name__ == '__main__':
    frames = [
        [[10,10], [20,15], [30,20]],
        [[11,10.5], [21,15.2], [30,20.3]],
        [[12,11],  [21.5,15.4]],
        [[13,11.5],[22,15.6]],
        [[14,12]]
    ]
    tracker = MultiObjectTrackerIMM6(dt=1.0, q_var=1.0, r_var=1.0)
    for idx, dets in enumerate(frames):
        print(f"Frame {idx} detections: {dets}")
        tracker.update(dets)
        for tr in tracker.get_tracks():
            print(
                f"  Track {tr['id']}: "
                f"px={tr['px']:.2f}, py={tr['py']:.2f}, "
                f"vx={tr['vx']:.2f}, vy={tr['vy']:.2f}, "
                f"ax={tr['ax']:.2f}, ay={tr['ay']:.2f}"
            )
        print()