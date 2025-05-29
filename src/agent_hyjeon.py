# frames_data_agent.py

import numpy as np
from kalman_filters import KalmanFilterCV6, KalmanFilterCA6, VariableTurnEKF, FixedTurnEKF
from frames_data import DT, NUM_FRAMES  # 기존 상수 재활용 :contentReference[oaicite:1]{index=1}



# 사용 예시
if __name__ == "__main__":
    configs = [
        {'model': KalmanFilterCV6, 'init': [0, 0, 1, 0.5, 0, 0]},
        {'model': KalmanFilterCA6, 'init': [5, 5, 0, 0, 0.2, -0.1]},
        {'model': VariableTurnEKF, 'init': [10, 0, 2, np.pi/4, 0.1, 0.05]},
    ]
    frames_agent = generate_from_agents(configs)
    # 이 frames_agent를 기존 tracker 업데이트나 visualizaiton에 그대로 연결!


    print(frames_agent)
