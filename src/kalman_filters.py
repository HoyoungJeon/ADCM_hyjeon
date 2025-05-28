import numpy as np
from scipy.stats import multivariate_normal

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

class VariableTurnEKF:
    """
    6D EKF for motion with **variable turn rate** (yaw_rate driven by yaw_acc):
      state x = [px, py, v, yaw, yaw_rate, yaw_acc]^T
      meas  z = [px, py]^T

    This model allows the turn rate (yaw_rate) to evolve over time via yaw_acc.
    """
    def __init__(self, dt=1.0, q_var=1.0, r_var=1.0):
        self.dt = dt

        # state: px, py, v, yaw, yaw_rate, yaw_acc
        self.x = np.zeros((6,1))
        self.P = np.eye(6)

        # process noise & measurement noise
        self.Q = q_var * np.eye(6)
        self.R = r_var * np.eye(2)

        # measurement matrix: we only observe px, py
        self.H = np.zeros((2,6))
        self.H[0,0] = 1
        self.H[1,1] = 1

    def predict(self):
        px, py, v, yaw, yaw_rate, yaw_acc = self.x.flatten()
        dt = self.dt

        # 1) non-linear state transition
        if abs(yaw_rate) > 1e-5:
            px_pred = px + (v/yaw_rate)*(np.sin(yaw + yaw_rate*dt) - np.sin(yaw))
            py_pred = py + (v/yaw_rate)*(-np.cos(yaw + yaw_rate*dt) + np.cos(yaw))
        else:
            # straight motion if yaw_rate ~ 0
            px_pred = px + v * np.cos(yaw) * dt
            py_pred = py + v * np.sin(yaw) * dt

        v_pred        = v
        yaw_pred      = yaw + yaw_rate*dt + 0.5*yaw_acc*dt*dt
        yaw_rate_pred = yaw_rate + yaw_acc*dt
        yaw_acc_pred  = yaw_acc

        self.x = np.array([
            [px_pred],
            [py_pred],
            [v_pred],
            [yaw_pred],
            [yaw_rate_pred],
            [yaw_acc_pred]
        ])

        # 2) compute Jacobian F (6×6)
        F = np.eye(6)
        # derivatives w.r.t. v, yaw, yaw_rate, yaw_acc
        if abs(yaw_rate) > 1e-5:
            F[0,2] = (1/yaw_rate)*(np.sin(yaw + yaw_rate*dt) - np.sin(yaw))
            F[1,2] = (1/yaw_rate)*(-np.cos(yaw + yaw_rate*dt) + np.cos(yaw))

            F[0,3] = (v/yaw_rate)*( np.cos(yaw + yaw_rate*dt) - np.cos(yaw))
            F[1,3] = (v/yaw_rate)*( np.sin(yaw + yaw_rate*dt) - np.sin(yaw))

            F[0,4] = (v/(yaw_rate**2))*( np.sin(yaw) - np.sin(yaw + yaw_rate*dt) ) \
                     + (v*dt/yaw_rate)*np.cos(yaw + yaw_rate*dt)
            F[1,4] = (v/(yaw_rate**2))*( np.cos(yaw + yaw_rate*dt) - np.cos(yaw) ) \
                     + (v*dt/yaw_rate)*np.sin(yaw + yaw_rate*dt)
        else:
            F[0,2] = np.cos(yaw)*dt
            F[1,2] = np.sin(yaw)*dt
            F[0,3] = -v*np.sin(yaw)*dt
            F[1,3] =  v*np.cos(yaw)*dt

        # yaw, yaw_rate, yaw_acc coupling
        F[3,4] = dt
        F[3,5] = 0.5 * dt * dt
        F[4,5] = dt

        # 3) covariance predict
        self.P = F @ self.P @ F.T + self.Q

        return self.x.copy()

    def update(self, z):
        """
        z: [px, py] measurement
        """
        z = np.array(z).reshape(2,1)

        # innovation
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # state update
        self.x = self.x + (K @ y)
        # covariance update
        I = np.eye(6)
        self.P = (I - K @ self.H) @ self.P

        return self.x.copy()

class FixedTurnEKF:
    """
    6D EKF for pure Constant Turn-Rate & Velocity (CTRV) model:
      state x = [px, py, v, yaw, yaw_rate, yaw_acc]^T
      meas  z = [px, py]^T

    This model assumes a constant turn rate (yaw_rate) over time.
    yaw_acc is carried only for IMM state‐dimension consistency and not used in prediction.
    """
    def __init__(self, dt=1.0, q_var=1.0, r_var=1.0):
        self.dt = dt
        # state and covariance
        self.x = np.zeros((6,1))
        self.P = np.eye(6)
        # process & measurement noise
        self.Q = q_var * np.eye(6)
        self.R = r_var * np.eye(2)
        # measurement matrix: only px, py observed
        self.H = np.zeros((2,6))
        self.H[0,0] = 1
        self.H[1,1] = 1

    def predict(self):
        px, py, v, yaw, yaw_rate, yaw_acc = self.x.flatten()
        dt = self.dt

        # 1) Nonlinear CTRV prediction
        if abs(yaw_rate) > 1e-5:
            px_pred = px + (v/yaw_rate)*(np.sin(yaw + yaw_rate*dt) - np.sin(yaw))
            py_pred = py + (v/yaw_rate)*(-np.cos(yaw + yaw_rate*dt) + np.cos(yaw))
        else:
            px_pred = px + v * np.cos(yaw) * dt
            py_pred = py + v * np.sin(yaw) * dt

        v_pred         = v
        yaw_pred       = yaw + yaw_rate * dt
        yaw_rate_pred  = yaw_rate
        yaw_acc_pred   = yaw_acc  # unchanged

        self.x = np.array([
            [px_pred],
            [py_pred],
            [v_pred],
            [yaw_pred],
            [yaw_rate_pred],
            [yaw_acc_pred]
        ])

        # 2) Jacobian F (6×6)
        F = np.eye(6)
        if abs(yaw_rate) > 1e-5:
            F[0,2] = (1/yaw_rate)*(np.sin(yaw + yaw_rate*dt) - np.sin(yaw))
            F[1,2] = (1/yaw_rate)*(-np.cos(yaw + yaw_rate*dt) + np.cos(yaw))
            F[0,3] = (v/yaw_rate)*(np.cos(yaw + yaw_rate*dt) - np.cos(yaw))
            F[1,3] = (v/yaw_rate)*(np.sin(yaw + yaw_rate*dt) - np.sin(yaw))
            F[0,4] = (v/(yaw_rate**2))*(np.sin(yaw) - np.sin(yaw + yaw_rate*dt)) \
                     + (v*dt/yaw_rate)*np.cos(yaw + yaw_rate*dt)
            F[1,4] = (v/(yaw_rate**2))*(np.cos(yaw + yaw_rate*dt) - np.cos(yaw)) \
                     + (v*dt/yaw_rate)*np.sin(yaw)
        else:
            F[0,2] = np.cos(yaw) * dt
            F[1,2] = np.sin(yaw) * dt
            F[0,3] = -v * np.sin(yaw) * dt
            F[1,3] =  v * np.cos(yaw) * dt

        # yaw ← yaw_rate coupling
        F[3,4] = dt
        # no coupling to yaw_acc (F[3,5] stays 0)
        # yaw_rate and yaw_acc states remain constant (diagonals = 1)

        # 3) Covariance predict
        self.P = F @ self.P @ F.T + self.Q
        return self.x.copy()

    def update(self, z):
        z = np.array(z).reshape(2,1)
        # innovation
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        # state & covariance update
        self.x = self.x + (K @ y)
        self.P = (np.eye(6) - K @ self.H) @ self.P
        return self.x.copy()



class IMMEstimator:
    """
    Interactive Multiple Model estimator combining several 6D Kalman filters.
    - models: 리스트 of KalmanFilterCV6, KalmanFilterCA6, CTRV6, CTRA6 인스턴스
    - PI: 모델 전이 확률 행렬 (MxM)
    """
    def __init__(self, models, PI=None):
        self.models = models
        self.M = len(models)
        # 초기 모델 확률: 균등분포
        self.mu = np.ones(self.M) / self.M
        # 모델 전이확률 행렬: 기본은 자주 바뀌지 않는다고 가정
        if PI is None:
            p_stay = 0.99
            p_switch = (1 - p_stay) / (self.M - 1)
            self.PI = np.full((self.M, self.M), p_switch)
            np.fill_diagonal(self.PI, p_stay)
        else:
            self.PI = PI

        self.x = self.models[0].x.copy()
        self.P = self.models[0].P.copy()


    def predict(self):
        # 1) Mixing probabilities
        c_j = self.PI.T @ self.mu            # 혼합계수 (M,)
        mu_ij = (self.PI * self.mu.reshape(-1,1)) / c_j.reshape(1,-1)  # (M,M)

        # 2) 혼합 상태(x0)와 혼합 공분산(P0)
        x0 = []
        P0 = []
        for j in range(self.M):
            x_mix = np.zeros_like(self.models[0].x)
            P_mix = np.zeros_like(self.models[0].P)
            for i in range(self.M):
                xi = self.models[i].x
                Pi = self.models[i].P
                w = mu_ij[i,j]
                x_mix += w * xi
            for i in range(self.M):
                xi = self.models[i].x
                Pi = self.models[i].P
                w = mu_ij[i,j]
                dx = xi - x_mix
                P_mix += w * (Pi + dx @ dx.T)
            x0.append(x_mix)
            P0.append(P_mix)

        # 3) 각 모델에 예측 단계 수행 (혼합 초기조건 사용)
        for j, model in enumerate(self.models):
            model.x = x0[j].copy()
            model.P = P0[j].copy()
            model.predict()

    def update(self, z):
        # 4) 각 모델 업데이트 및 likelihood 계산
        likelihoods = np.zeros(self.M)
        for j, model in enumerate(self.models):
            # 예측 잔차
            z_pred = model.H @ model.x
            S = model.H @ model.P @ model.H.T + model.R
            y = np.array(z).reshape(-1,1) - z_pred
            # 다변량 정규분포로 likelihood
            likelihoods[j] = multivariate_normal.pdf(
                y.flatten(), mean=np.zeros(y.size), cov=S
            )
            model.update(z)

        # 5) 모델 확률 갱신
        c = (likelihoods * (self.PI.T @ self.mu)).sum()
        self.mu = (likelihoods * (self.PI.T @ self.mu)) / c
        # 6) 결합 추정치 (가중합)
        # 상태
        x_comb = sum(self.mu[j] * self.models[j].x for j in range(self.M))
        # 공분산
        P_comb = np.zeros_like(self.models[0].P)
        for j in range(self.M):
            dx = self.models[j].x - x_comb
            P_comb += self.mu[j] * (self.models[j].P + dx @ dx.T)
        # 저장
        self.x = x_comb
        self.P = P_comb

    def get_state(self):
        # 결합 상태 반환 (flattened)
        return self.x.flatten()