import numpy as np

class KalmanFilterCV5:
    """
    5차원 상태 [x, y, v, yaw, yaw_rate]를 가지는
    Constant Velocity 모델.
    yaw, yaw_rate는 0으로 고정되어 회전 없는 직진만 모델링.
    """
    def __init__(self, dt, q_var, r_var):
        self.dt = dt
        self.F = np.array([
            [1, 0, dt, 0,  0],
            [0, 1,  0, 0,  0],
            [0, 0,  1, 0,  0],
            [0, 0,  0, 1, dt],
            [0, 0,  0, 0,  1],
        ])
        # 측정: z = [x, y, vx, vy]
        self.H = np.array([
            [1, 0, 0, 0, 0],  # x
            [0, 1, 0, 0, 0],  # y
            [0, 0, 1, 0, 0],  # vx ≈ v
            [0, 0, 0, 0, 0],  # vy ≈ 0
        ])
        self.R = r_var * np.diag([1, 1, 1, 1e-2])
        self.Q = q_var * np.eye(5)

        self.x = np.zeros((5,1))
        self.P = np.eye(5)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        y = z.reshape(4,1) - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(5) - K @ self.H) @ self.P


class ExtendedKalmanFilterCTRV:
    """
    5차원 상태 [x, y, v, yaw, yaw_rate]를 가지는 CTRV 모델.
    """
    def __init__(self, dt, q_var, r_var):
        self.dt = dt
        self.Q = q_var * np.eye(5)
        self.R = r_var * np.eye(4)
        self.x = np.zeros((5,1))
        self.P = np.eye(5)

    def f(self, x):
        px, py, v, yaw, yaw_rate = x.flatten()
        if abs(yaw_rate) < 1e-3:
            px += v * np.cos(yaw) * self.dt
            py += v * np.sin(yaw) * self.dt
        else:
            px += (v/yaw_rate) * (np.sin(yaw + yaw_rate*self.dt) - np.sin(yaw))
            py += (v/yaw_rate) * (-np.cos(yaw + yaw_rate*self.dt) + np.cos(yaw))
        yaw += yaw_rate * self.dt
        return np.array([[px], [py], [v], [yaw], [yaw_rate]])

    def jacobian_F(self, x):
        px, py, v, yaw, yaw_rate = x.flatten()
        dt = self.dt
        if abs(yaw_rate) < 1e-3:
            return np.array([
                [1, 0, np.cos(yaw)*dt, -v*np.sin(yaw)*dt, 0],
                [0, 1, np.sin(yaw)*dt,  v*np.cos(yaw)*dt, 0],
                [0, 0, 1,               0,                0],
                [0, 0, 0,               1,               dt],
                [0, 0, 0,               0,                1],
            ])
        s, c = np.sin(yaw), np.cos(yaw)
        dt_yaw = yaw_rate * dt
        s_dt = np.sin(yaw + dt_yaw)
        c_dt = np.cos(yaw + dt_yaw)
        return np.array([
            [1, 0, (s_dt - s)/yaw_rate,
             v*(c_dt*yaw_rate*dt - s_dt + s)/(yaw_rate**2),
             v*(c_dt*dt - (s_dt - s)/yaw_rate)/yaw_rate],
            [0, 1, (-c_dt + c)/yaw_rate,
             v*(s_dt*yaw_rate*dt + c_dt - c)/(yaw_rate**2),
             v*(s_dt*dt - (-c_dt + c)/yaw_rate)/yaw_rate],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, dt],
            [0, 0, 0, 0, 1],
        ])

    def get_H(self):
        px, py, v, yaw, _ = self.x.flatten()
        return np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, np.cos(yaw), -v*np.sin(yaw), 0],
            [0, 0, np.sin(yaw),  v*np.cos(yaw), 0],
        ])

    def predict(self):
        self.x = self.f(self.x)
        F_j = self.jacobian_F(self.x)
        self.P = F_j @ self.P @ F_j.T + self.Q

    def update(self, z):
        H = self.get_H()
        z_pred = H @ self.x
        y = z.reshape(4,1) - z_pred
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(5) - K @ H) @ self.P


class IMM:
    """
    filters: list of filter instances
    trans_mat: 모델 전이확률 행렬 Λ (NxN)
    init_probs: 초기 모델 확률 벡터 (N)
    """
    def __init__(self, filters, trans_mat, init_probs):
        self.filters = filters
        self.P_ij = np.array(trans_mat)
        self.mu   = np.array(init_probs)
        self.M    = len(filters)

    def predict_and_update(self, z):
        # 1) Mixing probabilities
        c_j   = self.P_ij.T @ self.mu
        mu_ij = (self.P_ij * self.mu.reshape(-1,1)) / c_j.reshape(1,-1)

        # 2) Mix states & covariances
        for j in range(self.M):
            x0 = sum(mu_ij[i,j] * self.filters[i].x for i in range(self.M))
            P0 = sum(mu_ij[i,j] * (
                     self.filters[i].P +
                     (self.filters[i].x - x0) @ (self.filters[i].x - x0).T
                 ) for i in range(self.M))
            self.filters[j].x = x0
            self.filters[j].P = P0

        # 3) Predict & update + likelihood
        likelihoods = np.zeros(self.M)
        for j, f in enumerate(self.filters):
            f.predict()
            # get measurement jacobian H
            H = f.H if hasattr(f, 'H') else f.get_H()
            z_pred = H @ f.x
            y      = z.reshape(4,1) - z_pred
            S      = H @ f.P @ H.T + f.R
            exponent = -0.5 * float((y.T @ np.linalg.inv(S) @ y))
            norm     = np.sqrt(((2*np.pi)**4) * np.linalg.det(S))
            likelihoods[j] = float(np.exp(exponent) / norm)
            f.update(z)

        # 4) Model probability update
        self.mu = likelihoods * c_j
        self.mu /= np.sum(self.mu)

        # 5) Combine state & covariance
        x_comb = sum(self.mu[j] * self.filters[j].x for j in range(self.M))
        P_comb = sum(self.mu[j] * (
                 self.filters[j].P +
                 (self.filters[j].x - x_comb) @ (self.filters[j].x - x_comb).T
             ) for j in range(self.M))

        return x_comb, P_comb


if __name__ == "__main__":
    dt = 0.1
    kf_cv5   = KalmanFilterCV5(dt, q_var=1.0, r_var=0.5)
    ekf_ctrv = ExtendedKalmanFilterCTRV(dt, q_var=1.0, r_var=0.5)

    imm = IMM(
        filters=[kf_cv5, ekf_ctrv],
        trans_mat=[[0.95, 0.05],
                   [0.05, 0.95]],
        init_probs=[0.5, 0.5]
    )

    measurements = [
        np.array([10 + i*0.1, 5 + i*0.05, 1.0, 0.2])
        for i in range(20)
    ]

    for idx, z in enumerate(measurements):
        x_est, P_est = imm.predict_and_update(z)
        print(f"Step {idx:02d} | 예측 상태: {x_est.flatten()}")
        print(f"          | 모델 확률: {imm.mu}")
        print("-" * 60)
