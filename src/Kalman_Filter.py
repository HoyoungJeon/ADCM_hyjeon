import numpy as np

class KalmanFilterCV:
    def __init__(self,
                 process_noise_std: float = 1.0,
                 measurement_noise_std: float = 0.5,
                 initial_P: float = 500.0):
        # 측정 모델: z = [px, py]
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        r = measurement_noise_std
        self.R = np.eye(2) * (r**2)

        # 상태 벡터 [px, py, vx, vy]
        self.x = np.zeros((4,1))
        # 공분산
        self.P = np.eye(4) * initial_P

    def predict(self, dt: float, process_noise_std: float):
        # 상태전이
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1,  0],
                      [0, 0, 0,  1]])
        # 프로세스 잡음 (constant accel 가정)
        q = process_noise_std
        Q = q**2 * np.array([
            [dt**4/4,      0, dt**3/2,      0],
            [     0, dt**4/4,      0, dt**3/2],
            [dt**3/2,      0,   dt**2,      0],
            [     0, dt**3/2,      0,   dt**2]
        ])

        self.x = F.dot(self.x)
        self.P = F.dot(self.P).dot(F.T) + Q

    def update(self, z: np.ndarray):
        z = z.reshape((2,1))
        y = z - self.H.dot(self.x)
        S = self.H.dot(self.P).dot(self.H.T) + self.R
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))
        self.x = self.x + K.dot(y)
        I = np.eye(4)
        self.P = (I - K.dot(self.H)).dot(self.P)

    def current_state(self):
        return self.x.flatten()  # px, py, vx, vy


if __name__ == "__main__":
    # — 합성 데이터 생성 (20포인트) —
    np.random.seed(42)
    num_points = 20
    dt = 0.1
    times = np.arange(0, num_points * dt, dt)  # 0.0,0.1,…,1.9
    true_vx, true_vy = 1.0, 0.5
    true_positions = np.vstack((true_vx * times,
                                true_vy * times)).T
    meas_noise_std = 0.1
    meas_pos = [
        tuple(pos + np.random.normal(0, meas_noise_std, size=2))
        for pos in true_positions
    ]

    # — 필터 초기화 —
    kf = KalmanFilterCV(process_noise_std=1.0,
                        measurement_noise_std=meas_noise_std,
                        initial_P=500.0)

    # 헤더 출력
    print(f"{'t':>4} | {'meas_x':>7} {'meas_y':>7} | "
          f"{'pred_x':>7} {'pred_y':>7} | "
          f"{'est_x':>7} {'est_y':>7} | "
          f"{'vx':>6} {'vy':>6}")
    print("-"*70)

    # 첫 타임스탬프 설정
    t_prev = times[0]
    # 첫 측정값으로만 초기 상태(px,py) 설정
    kf.x[:2] = np.array(meas_pos[0]).reshape(2,1)

    # 루프: predict → print(predicted) → update → print(estimated)
    for t, pos in zip(times[1:], meas_pos[1:]):
        dt_step = t - t_prev

        # 1) 예측
        kf.predict(dt=dt_step, process_noise_std=1.0)
        px_p, py_p, _, _ = kf.current_state()

        # 2) 측정 업데이트
        kf.update(np.array(pos))
        px_e, py_e, vx_e, vy_e = kf.current_state()

        # 3) 출력
        print(f"{t:4.2f} | "
              f"{pos[0]:7.3f} {pos[1]:7.3f} | "
              f"{px_p:7.3f} {py_p:7.3f} | "
              f"{px_e:7.3f} {py_e:7.3f} | "
              f"{vx_e:6.3f} {vy_e:6.3f}")

        t_prev = t
