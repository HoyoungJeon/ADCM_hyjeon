import numpy as np
from Kalman_Filter import KalmanFilterCV


def run_collision_simulation(
    ego_history: np.ndarray,
    target_history: np.ndarray,
    times: np.ndarray,
    process_noise_std: float = 1.0,
    meas_noise_std: float = 0.1,
    predict_horizon: float = 3.0,
    predict_dt: float = 0.1,
    collision_dist: float = 2.0,
) -> float:
    """
    하나의 시나리오, 즉 과거 측정 이력을 주면
    - 칼만 필터로 속도/위치 추정
    - 향후 궤적 예측
    - 충돌 예상 시 t_future 리턴 (없으면 None)
    """
    # 1) 칼만 필터 초기화
    ego_kf = KalmanFilterCV(process_noise_std, meas_noise_std)
    tgt_kf = KalmanFilterCV(process_noise_std, meas_noise_std)

    # 2) 초기 상태 설정
    ego_kf.x[:2] = ego_history[0].reshape(2, 1)
    tgt_kf.x[:2] = target_history[0].reshape(2, 1)

    # 3) 과거 측정 업데이트
    t_prev = times[0]
    for t, (e_m, t_m) in zip(times[1:], zip(ego_history[1:], target_history[1:])):
        dt = t - t_prev
        ego_kf.predict(dt, process_noise_std)
        ego_kf.update(e_m)
        tgt_kf.predict(dt, process_noise_std)
        tgt_kf.update(t_m)
        t_prev = t

    # 4) 추정된 현재 상태
    ego_px, ego_py, ego_vx, ego_vy = ego_kf.current_state()
    tgt_px, tgt_py, tgt_vx, tgt_vy = tgt_kf.current_state()

    # 5) 미래 예측
    t_future = predict_dt
    while t_future <= predict_horizon:
        ex = ego_px + ego_vx * t_future
        ey = ego_py + ego_vy * t_future
        tx = tgt_px + tgt_vx * t_future
        ty = tgt_py + tgt_vy * t_future

        if np.hypot(ex - tx, ey - ty) < collision_dist:
            return t_future
        t_future += predict_dt

    return None


def main():
    # 공통 파라미터
    dt = 0.1
    history_times = np.arange(-0.4, dt/2, dt)   # [-0.4, -0.3, ..., 0.0]

    # 시나리오들 정의: (ego_history, target_history, 설명)
    scenarios = [
        {
            "name": "직진 vs 좌회전",
            "ego": np.column_stack((np.zeros(5), np.linspace(-24, -12, 5))),
            "tgt": np.column_stack((np.linspace(-24, -12, 5), np.zeros(5))),
        },
        {
            "name": "직진 vs 우회전",
            "ego": np.column_stack((np.zeros(5), np.linspace(-24, -12, 5))),
            "tgt": np.column_stack((np.linspace( 24,  12, 5), np.zeros(5))),
        },
        {
            "name": "대각선 충돌 코너 케이스",
            "ego": np.column_stack((np.linspace(-24, -12, 5), np.linspace(-24, -12, 5))),
            "tgt": np.column_stack((np.linspace(-24, -12, 5), np.linspace( 24,  12, 5))),
        },
        # 여기에 원하는 만큼 추가...
    ]

    for sc in scenarios:
        t_col = run_collision_simulation(
            ego_history   = sc["ego"],
            target_history= sc["tgt"],
            times         = history_times
        )
        if t_col is not None:
            print(f"{sc['name']}: 충돌 예상 t+{t_col:.1f}s")
        else:
            print(f"{sc['name']}: 충돌 없음")


if __name__ == "__main__":
    main()
