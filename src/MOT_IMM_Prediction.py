# Updated MultiObjectTracker with 6-D State IMM (CV6, CA, CTRV6, CTRA)

import numpy as np
import copy
from scipy.optimize import linear_sum_assignment

# --- Kalman Filter Models with 6-D State ---

class KalmanFilterCV6:
    """Constant Velocity with 6D state [x, y, vx, vy, yaw, yaw_rate]"""
    def __init__(self, dt=1.0, q_var=1.0, r_var=1.0):
        self.dt = dt
        # State transition (6x6): yaw and yaw_rate remain
        self.F = np.array([
            [1, 0, dt, 0,  0,   0],
            [0, 1,  0, dt, 0,   0],
            [0, 0,  1,  0, 0,   0],
            [0, 0,  0,  1, 0,   0],
            [0, 0,  0,  0, 1,  dt],
            [0, 0,  0,  0, 0,   1],
        ])
        # Measure x,y only
        self.H = np.zeros((2,6)); self.H[0,0]=1; self.H[1,1]=1
        self.Q = q_var * np.eye(6)
        self.R = r_var * np.eye(2)
        self.x = np.zeros((6,1))
        self.P = np.eye(6)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        z = z.reshape(2,1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P


class KalmanFilterCA6:
    """Constant Acceleration with 6D state [x, y, vx, vy, ax, ay]"""
    def __init__(self, dt=1.0, q_var=1.0, r_var=1.0):
        self.dt = dt
        self.F = np.array([
            [1,0,dt,0,0.5*dt*dt, 0],
            [0,1,0,dt,0,0.5*dt*dt],
            [0,0,1,0,dt,0],
            [0,0,0,1,0,dt],
            [0,0,0,0,1,0],
            [0,0,0,0,0,1]
        ])
        self.H = np.zeros((2,6)); self.H[0,0]=1; self.H[1,1]=1
        self.Q = q_var * np.eye(6)
        self.R = r_var * np.eye(2)
        self.x = np.zeros((6,1))
        self.P = np.eye(6)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        z = z.reshape(2,1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P


class ExtendedKalmanFilterCTRV6:
    """CTRV extended to 6D: [x, y, v, yaw, yaw_rate, a] with a fixed at 0"""
    def __init__(self, dt=1.0, q_var=1.0, r_var=1.0):
        self.dt = dt
        self.Q = q_var * np.eye(6)
        self.R = r_var * np.eye(2)
        self.x = np.zeros((6,1))
        self.P = np.eye(6)

    def f(self, x):
        px, py, v, yaw, yaw_rate, a = x.flatten()
        # keep a=0 for CTRV
        if abs(yaw_rate) < 1e-3:
            px += v * np.cos(yaw) * self.dt
            py += v * np.sin(yaw) * self.dt
        else:
            px += (v/yaw_rate)*(np.sin(yaw+yaw_rate*self.dt)-np.sin(yaw))
            py += (v/yaw_rate)*(-np.cos(yaw+yaw_rate*self.dt)+np.cos(yaw))
        yaw += yaw_rate*self.dt
        return np.array([[px],[py],[v],[yaw],[yaw_rate],[0]])

    def predict(self):
        self.x = self.f(self.x)
        F = np.eye(6)  # approximate Jacobian sector
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        H = np.zeros((2,6)); H[0,0]=1; H[1,1]=1
        z = z.reshape(2,1)
        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (np.eye(6) - K @ H) @ self.P


class ExtendedKalmanFilterCTRA6:
    """CTRA with 6D state [x, y, v, yaw, yaw_rate, a]"""
    def __init__(self, dt=1.0, q_var=1.0, r_var=1.0):
        self.dt = dt
        self.Q = q_var * np.eye(6)
        self.R = r_var * np.eye(2)
        self.x = np.zeros((6,1))
        self.P = np.eye(6)

    def f(self, x):
        px, py, v, yaw, yaw_rate, a = x.flatten()
        v_new = v + a*self.dt
        if abs(yaw_rate) < 1e-3:
            px += v_new*np.cos(yaw)*self.dt
            py += v_new*np.sin(yaw)*self.dt
        else:
            px += (v_new/yaw_rate)*(np.sin(yaw+yaw_rate*self.dt)-np.sin(yaw))
            py += (v_new/yaw_rate)*(-np.cos(yaw+yaw_rate*self.dt)+np.cos(yaw))
        yaw += yaw_rate*self.dt
        return np.array([[px],[py],[v_new],[yaw],[yaw_rate],[a]])

    def predict(self):
        self.x = self.f(self.x)
        F = np.eye(6)
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        H = np.zeros((2,6)); H[0,0]=1; H[1,1]=1
        z = z.reshape(2,1)
        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (np.eye(6) - K @ H) @ self.P

# --- IMM framework (unchanged) ---

class IMM:
    def __init__(self, filters, trans_mat, init_probs):
        self.filters = filters
        self.P_ij = np.array(trans_mat)
        self.mu = np.array(init_probs)
        self.M = len(filters)

    def predict_update(self, z):
        c_j = self.P_ij.T @ self.mu
        mu_ij = (self.P_ij * self.mu.reshape(-1,1)) / c_j.reshape(1,-1)
        for j in range(self.M):
            x0 = sum(mu_ij[i,j]*self.filters[i].x for i in range(self.M))
            P0 = sum(mu_ij[i,j]*(self.filters[i].P + (self.filters[i].x-x0)@(self.filters[i].x-x0).T)
                     for i in range(self.M))
            self.filters[j].x, self.filters[j].P = x0, P0
        likelihood = np.zeros(self.M)
        for j,f in enumerate(self.filters):
            f.predict()
            f.update(z)
            H = f.H if hasattr(f,'H') else np.zeros((2,6)); H[0,0]=1; H[1,1]=1
            res = z.reshape(2,1) - H @ f.x
            S = H @ f.P @ H.T + f.R
            expo = -0.5 * float((res.T @ np.linalg.inv(S) @ res).item())
            norm = np.sqrt((2 * np.pi) ** 2 * np.linalg.det(S))
            likelihood[j] = float(np.exp(expo) / norm)
        self.mu = (likelihood.flatten()*c_j)
        self.mu /= np.sum(self.mu)
        x_comb = sum(self.mu[j]*self.filters[j].x for j in range(self.M))
        P_comb = sum(self.mu[j]*(self.filters[j].P + (self.filters[j].x-x_comb)@(self.filters[j].x-x_comb).T)
                     for j in range(self.M))
        return x_comb, P_comb

# --- MultiObjectTracker with IMM ---

class Track:
    def __init__(self, detection, track_id, dt=1.0):
        self.id = track_id
        self.skipped = 0
        det = np.array(detection)
        # init filters with 6D state
        cv6 = KalmanFilterCV6(dt);    cv6.x[0:2,0] = det
        ca6 = KalmanFilterCA6(dt);    ca6.x[0:2,0] = det
        ctrv6 = ExtendedKalmanFilterCTRV6(dt); ctrv6.x[0:2,0] = det
        ctra6 = ExtendedKalmanFilterCTRA6(dt); ctra6.x[0:2,0] = det
        self.imm = IMM(
            [cv6, ca6, ctrv6, ctra6],
            trans_mat=[[0.9,0.03,0.04,0.03]]*4,
            init_probs=[0.25]*4
        )
        self.detect = det

    def predict_update(self, detection):
        self.detect = np.array(detection)
        x, P = self.imm.predict_update(self.detect)
        self.skipped = 0
        return x, P

    def miss(self):
        self.skipped += 1

class MultiObjectTracker:
    def __init__(self, max_skipped=5, dist_threshold=50.0):
        self.tracks = []
        self.next_id = 0
        self.max_skipped = max_skipped
        self.dist_threshold = dist_threshold

    @staticmethod
    def distance(a, b):
        return np.linalg.norm(np.array(a)-np.array(b))

    def associate(self, detections):
        N, M = len(self.tracks), len(detections)
        if N == 0:
            return [], list(range(M)), []
        cost = np.zeros((N, M))
        for i, trk in enumerate(self.tracks):
            pred, _ = trk.imm.predict_update(trk.detect)
            for j, det in enumerate(detections):
                cost[i, j] = self.distance(pred[:2].flatten(), det)
        row, col = linear_sum_assignment(cost)
        matched, u_det, u_trk = [], [], []
        for i, j in zip(row, col):
            if cost[i, j] > self.dist_threshold:
                u_trk.append(i)
                u_det.append(j)
            else:
                matched.append((i, j))
        u_trk += [i for i in range(N) if i not in row]
        u_det += [j for j in range(M) if j not in col]
        return matched, u_det, u_trk

    def update(self, detections):
        matched, u_det, u_trk = self.associate(detections)
        for i, j in matched:
            self.tracks[i].predict_update(detections[j])
        for j in u_det:
            self.tracks.append(Track(detections[j], self.next_id))
            self.next_id += 1
        for i in u_trk:
            self.tracks[i].miss()
        self.tracks = [t for t in self.tracks if t.skipped <= self.max_skipped]

    def get_tracks(self):
        out = []
        for t in self.tracks:
            x, _ = t.imm.predict_update(t.detect)
            out.append({
                'id': t.id,
                'x': round(float(x[0,0]), 1),
                'y': round(float(x[1,0]), 1),
                'vx': round(float(x[2,0]), 1),
                'vy': round(float(x[3,0]), 1),
            })
        return out

def predict_future_states(track, steps=5):
    """
    Given a Track with an IMM instance, predict the next `steps` future states [x, y].
    We perform the IMM 'mixing & predict' cycles without measurement updates.
    """
    # Deep copy the IMM to avoid altering the real filter
    imm = copy.deepcopy(track.imm)
    # Copy current model probabilities
    mu = imm.mu.copy()

    future_positions = []
    for _ in range(steps):
        # 1) Mixing probabilities
        c_j = imm.P_ij.T @ mu
        mu_ij = (imm.P_ij * mu.reshape(-1, 1)) / c_j.reshape(1, -1)

        # 2) Mix states and covariances
        for j in range(imm.M):
            # Mix state
            x0 = sum(mu_ij[i, j] * imm.filters[i].x for i in range(imm.M))
            # Mix covariance
            P0 = sum(mu_ij[i, j] * (imm.filters[i].P +
                     (imm.filters[i].x - x0) @ (imm.filters[i].x - x0).T)
                     for i in range(imm.M))
            imm.filters[j].x = x0
            imm.filters[j].P = P0

        # 3) Predict step for each filter
        for f in imm.filters:
            f.predict()

        # 4) Update model probabilities (no new measurement, so only transition)
        mu = c_j / np.sum(c_j)

        # 5) Combine states weighted by updated mu
        x_comb = sum(mu[j] * imm.filters[j].x for j in range(imm.M))
        # Collect the position part
        future_positions.append((float(x_comb[0, 0]), float(x_comb[1, 0])))

    return future_positions


def simulate_and_predict():
    mot = MultiObjectTracker()
    # Simulate 20 frames of a single object moving diagonally
    frames = [[10 + i * 0.5, 10 + i * 0.3] for i in range(20)]

    print("=== Simulation of 20 Frames ===")
    for idx, det in enumerate(frames):
        mot.update([det])  # single detection per frame
        tracks = mot.get_tracks()
        # Print detection and current track state
        print(f"Frame {idx:02d}: Detection=({det[0]:.1f},{det[1]:.1f})", end='  ')
        if tracks:
            trk = tracks[0]
            print(f"Track ID={trk['id']}, Pos=({trk['x']:.1f},{trk['y']:.1f}), Vel=({trk['vx']:.1f},{trk['vy']:.1f})")
        else:
            print("No track yet")

    # Predict next 5 future positions for the first track
    if mot.tracks:
        future = predict_future_states(mot.tracks[0], steps=5)
        print("\nPredicted future positions (next 5 steps):")
        for step, pos in enumerate(future, 1):
            print(f"  Step +{step}: ({pos[0]:.1f}, {pos[1]:.1f})")


# Example usage:
if __name__ == '__main__':
    simulate_and_predict()