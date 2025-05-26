import numpy as np


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
