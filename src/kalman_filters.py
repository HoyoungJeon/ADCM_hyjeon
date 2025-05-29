import numpy as np
from scipy.stats import multivariate_normal

class KalmanFilterCV6:
    """Constant Velocity model in 6D (ax = ay = 0)."""
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
        # Measurement matrix: we observe px and py
        self.H = np.zeros((2,6))
        self.H[0,0] = 1
        self.H[1,1] = 1
        # Process and measurement noise covariances
        self.Q = q_var * np.eye(6)
        self.R = r_var * np.eye(2)
        # Initial state and covariance
        self.x = np.zeros((6,1))
        self.P = np.eye(6)

    def predict(self):
        """Predict step: x = F x; P = F P Fᵀ + Q."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z):
        """Update step with measurement z = [px, py]."""
        z = np.array(z).reshape(2,1)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        return self.x.copy()


class KalmanFilterCA6:
    """Constant Acceleration model in 6D (ax, ay included)."""
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
        # We observe px and py
        self.H = np.zeros((2,6))
        self.H[0,0] = 1
        self.H[1,1] = 1
        # Noise covariances
        self.Q = q_var * np.eye(6)
        self.R = r_var * np.eye(2)
        # Initial state and covariance
        self.x = np.zeros((6,1))
        self.P = np.eye(6)

    def predict(self):
        """Predict step: x = F x; P = F P Fᵀ + Q."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z):
        """Update step with measurement z = [px, py]."""
        z = np.array(z).reshape(2,1)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        return self.x.copy()


class VariableTurnEKF:
    """
    Extended Kalman Filter with variable turn-rate (CTRA):
      state x = [px, py, v, yaw, yaw_rate, yaw_acc]^T
      measurement z = [px, py]^T

    The turn rate (yaw_rate) is driven by yaw_acc over time.
    """
    def __init__(self, dt=1.0, q_var=1.0, r_var=1.0):
        self.dt = dt
        # State and covariance
        self.x = np.zeros((6,1))
        self.P = np.eye(6)
        # Noise covariances
        self.Q = q_var * np.eye(6)
        self.R = r_var * np.eye(2)
        # We observe px and py
        self.H = np.zeros((2,6))
        self.H[0,0] = 1
        self.H[1,1] = 1

    def predict(self):
        """Nonlinear predict and covariance update."""
        px, py, v, yaw, yaw_rate, yaw_acc = self.x.flatten()
        dt = self.dt

        # 1) Nonlinear state propagation
        if abs(yaw_rate) > 1e-5:
            px_pred = px + (v/yaw_rate)*(np.sin(yaw + yaw_rate*dt) - np.sin(yaw))
            py_pred = py + (v/yaw_rate)*(-np.cos(yaw + yaw_rate*dt) + np.cos(yaw))
        else:
            # Straight-line motion if yaw_rate ≈ 0
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

        # 2) Compute Jacobian F (6×6)
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
            F[0,2] = np.cos(yaw)*dt
            F[1,2] = np.sin(yaw)*dt
            F[0,3] = -v*np.sin(yaw)*dt
            F[1,3] =  v*np.cos(yaw)*dt

        # Coupling between yaw, yaw_rate, yaw_acc
        F[3,4] = dt
        F[3,5] = 0.5 * dt * dt
        F[4,5] = dt

        # 3) Covariance predict
        self.P = F @ self.P @ F.T + self.Q
        return self.x.copy()

    def update(self, z):
        """EKF update with measurement z = [px, py]."""
        z = np.array(z).reshape(2,1)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        return self.x.copy()


class FixedTurnEKF:
    """
    Extended Kalman Filter for CTRV (constant turn-rate):
      state x = [px, py, v, yaw, yaw_rate, yaw_acc]^T
      measurement z = [px, py]^T

    yaw_acc is included for IMM state consistency but is not used in prediction.
    """
    def __init__(self, dt=1.0, q_var=1.0, r_var=1.0):
        self.dt = dt
        self.x = np.zeros((6,1))
        self.P = np.eye(6)
        self.Q = q_var * np.eye(6)
        self.R = r_var * np.eye(2)
        self.H = np.zeros((2,6))
        self.H[0,0] = 1
        self.H[1,1] = 1

    def predict(self):
        """Nonlinear CTRV predict and covariance update."""
        px, py, v, yaw, yaw_rate, yaw_acc = self.x.flatten()
        dt = self.dt

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

        # Compute Jacobian F
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
            F[0,2] = np.cos(yaw)*dt
            F[1,2] = np.sin(yaw)*dt
            F[0,3] = -v*np.sin(yaw)*dt
            F[1,3] =  v*np.cos(yaw)*dt

        # yaw ← yaw_rate coupling
        F[3,4] = dt
        # no coupling to yaw_acc

        # Covariance predict
        self.P = F @ self.P @ F.T + self.Q
        return self.x.copy()

    def update(self, z):
        """EKF update with measurement z = [px, py]."""
        z = np.array(z).reshape(2,1)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        return self.x.copy()


class IMMEstimator:
    """
    Interactive Multiple Model (IMM) estimator combining multiple 6D filters.
    - models: list of individual Kalman/EKF filter instances
    - PI: model transition probability matrix (MxM)
    """
    def __init__(self, models, PI=None):
        self.models = models
        self.M = len(models)
        # Initial model probabilities: uniform
        self.mu = np.ones(self.M) / self.M

        # Transition probability matrix (assume high self-transition by default)
        if PI is None:
            p_stay = 0.8
            p_switch = (1 - p_stay) / (self.M - 1)
            self.PI = np.full((self.M, self.M), p_switch)
            np.fill_diagonal(self.PI, p_stay)
        else:
            self.PI = PI

        # Initialize combined state and covariance from the first model
        self.x = self.models[0].x.copy()
        self.P = self.models[0].P.copy()

    def predict(self):
        """IMM predict step: mixing, individual predicts."""
        # 1) Mixing probabilities c_j and mu_{i→j}
        c_j = self.PI.T @ self.mu                  # (M,)
        mu_ij = (self.PI * self.mu.reshape(-1,1)) / c_j.reshape(1,-1)  # (M×M)

        # 2) Mixed initial states x0[j] and covariances P0[j]
        x0, P0 = [], []
        for j in range(self.M):
            # Mix states
            x_mix = sum(mu_ij[i,j] * self.models[i].x for i in range(self.M))
            # Mix covariances
            P_mix = np.zeros_like(self.models[0].P)
            for i in range(self.M):
                dx = self.models[i].x - x_mix
                P_mix += mu_ij[i,j] * (self.models[i].P + dx @ dx.T)
            x0.append(x_mix)
            P0.append(P_mix)

        # 3) Each model uses its mixed initial condition for predict
        for j, model in enumerate(self.models):
            model.x = x0[j].copy()
            model.P = P0[j].copy()
            model.predict()

    def update(self, z):
        """IMM update: individual updates, likelihoods, model probability update, combine."""
        # 4) Update each model and compute its likelihood
        likelihoods = np.zeros(self.M)
        for j, model in enumerate(self.models):
            # Innovation
            z_pred = model.H @ model.x
            S = model.H @ model.P @ model.H.T + model.R
            y = np.array(z).reshape(-1,1) - z_pred
            # Multivariate normal likelihood
            likelihoods[j] = multivariate_normal.pdf(
                y.flatten(), mean=np.zeros(y.size), cov=S
            )
            model.update(z)

        # 5) Update model probabilities
        c = (likelihoods * (self.PI.T @ self.mu)).sum()
        self.mu = (likelihoods * (self.PI.T @ self.mu)) / c

        # 6) Compute combined state x and covariance P
        x_comb = sum(self.mu[j] * self.models[j].x for j in range(self.M))
        P_comb = np.zeros_like(self.models[0].P)
        for j in range(self.M):
            dx = self.models[j].x - x_comb
            P_comb += self.mu[j] * (self.models[j].P + dx @ dx.T)

        # Store combined estimates
        self.x = x_comb
        self.P = P_comb

    def get_state(self):
        """Return the combined IMM state as a flat array [px,py,v,yaw,yaw_rate,yaw_acc]."""
        return self.x.flatten()
