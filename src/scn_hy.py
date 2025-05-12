import numpy as np
from Kalman_Filter import KalmanFilterCV


def simulate_collision_prediction():
    """Simulate a simple intersection scenario and predict collisions."""
    # Past measurements (dt = 0.1s, 5 points)
    times = [-0.4, -0.3, -0.2, -0.1, 0.0]
    ego_measurements = [
        (0, -24),
        (0, -21),
        (0, -18),
        (0, -15),
        (0, -12),
    ]
    target_measurements = [
        (-24, 0),
        (-21, 0),
        (-18, 0),
        (-15, 0),
        (-12, 0),
    ]

    ego_kf = KalmanFilterCV(process_noise_std=1.0,
                            measurement_noise_std=0.1,
                            initial_P=500.0)
    target_kf = KalmanFilterCV(process_noise_std=1.0,
                               measurement_noise_std=0.1,
                               initial_P=500.0)

    # Initialize state with first measurement
    ego_kf.x[:2] = np.array(ego_measurements[0]).reshape(2, 1)
    target_kf.x[:2] = np.array(target_measurements[0]).reshape(2, 1)

    # Update filter with past measurements
    previous_time = times[0]
    for t, (ego_pos, target_pos) in zip(
        times[1:], zip(ego_measurements[1:], target_measurements[1:])
    ):
        dt = t - previous_time
        ego_kf.predict(dt, process_noise_std=1.0)
        ego_kf.update(np.array(ego_pos))
        target_kf.predict(dt, process_noise_std=1.0)
        target_kf.update(np.array(target_pos))
        previous_time = t

    # Extract estimated state
    ego_px, ego_py, ego_vx, ego_vy = ego_kf.current_state()
    target_px, target_py, target_vx, target_vy = target_kf.current_state()

    # Future collision prediction
    time_horizon = 3.0  # seconds
    time_step = 0.1     # seconds
    collision_detected = False

    t_future = time_step
    while t_future <= time_horizon:
        # Predict future positions
        ego_x = ego_px + ego_vx * t_future
        ego_y = ego_py + ego_vy * t_future
        target_x = target_px + target_vx * t_future
        target_y = target_py + target_vy * t_future

        distance = np.hypot(ego_x - target_x, ego_y - target_y)
        if distance < 2.0:
            print(f"Collision predicted at t+{t_future:.1f}s, distance={distance:.2f}m")
            collision_detected = True
            break

        t_future += time_step

    if not collision_detected:
        print("No collision predicted within the time horizon.")


if __name__ == "__main__":
    simulate_collision_prediction()

