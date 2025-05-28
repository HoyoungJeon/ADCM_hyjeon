import math
import numpy as np

# ── 공통 파라미터 ──
DT         = 0.1        # 프레임 간 시간 간격 (초)
NUM_FRAMES = 20         # 프레임 수

# ── 1. 고정 예제 시퀀스 ──
frames1 = [
    # Frame 0–2: A, C
    [[0.0, 0.0], [5.0, 0.0]],
    [[1.0, 0.5], [5.0, 1.0]],
    [[2.0, 1.0], [5.0, 2.0]],
    # Frame 3–5: A, B, C
    [[3.0, 1.5], [2.0, 5.0], [5.0, 3.0]],
    [[4.0, 2.0], [3.0, 5.0], [5.0, 4.0]],
    [[5.0, 2.5], [4.0, 5.0], [5.0, 5.0]],
    # Frame 6–8: A, B
    [[6.0, 3.0], [5.0, 5.0]],
    [[7.0, 3.5], [6.0, 5.0]],
    [[8.0, 4.0], [7.0, 5.0]],
    # Frame 9–10: A only
    [[9.0, 4.5]],
    [[10.0, 5.0]],
    # Frame 11: empty
    []
]

frames_cross = [
    [[0, 0],   [10, 0]],
    [[1, 1],   [9, -1]],
    [[2, 2],   [8, -2]],
    [[3, 3],   [7, -3]],  # 교차점 통과
    [[4, 4],   [6, -4]],
    [[5, 5],   [5, -5]],
    [[6, 6],   [4, -6]],
]

frames_entry_exit = [
    [],                 # 0: 아무도 없음
    [[0, 0]],           # 1: A 등장
    [[1, 0]],           # 2
    [[2, 0], [5, 5]],   # 3: B 등장
    [[3, 0], [6, 5]],   # 4
    [[4, 0], [7, 5], [10, 10]],  # 5: C 등장
    [[5, 0], [8, 5], [11, 10]],
    [[6, 0], [9, 5]],   # 7: C 퇴장
    [[7, 0]],           # 8: B 퇴장
    [[8, 0]],           # 9
    [],                 # 10: A 퇴장
]

# ── 2. 함수 기반 생성 시퀀스 ──
def generate_sine_vs_line(n_frames=NUM_FRAMES, x_step=0.5, amplitude=2, freq=0.3, line_speed=0.5):
    """
    A: x 증가 → y = amplitude * sin(freq * x)
    B: 일정 속도로 x 증가, y = 0
    """
    frames = []
    for i in range(n_frames):
        x = round(i * x_step, 2)
        y_sine = round(amplitude * math.sin(freq * x), 2)
        y_line = 0.0
        frames.append([[x, y_sine], [x, y_line]])
    return frames

frames_sine = generate_sine_vs_line()

def generate_unit_semicircle(steps=100):
    """
    0부터 π까지 steps만큼 나눠 반원 궤적을 생성.
    """
    theta = np.linspace(0, math.pi, steps)
    return [[[float(np.cos(t)), float(np.sin(t))]] for t in theta]

frames_random = generate_unit_semicircle()

# ── 3. 물리 모델 기반 생성 시퀀스 ──
def generate_ctrv(radius=5.0, n_frames=NUM_FRAMES, dt=DT):
    """CTRV6 모델에 최적화된 등속 원운동 궤적."""
    T_total = (n_frames - 1) * dt
    omega = 2 * math.pi / T_total
    return [
        [[round(radius * math.cos(omega * i * dt), 2),
          round(radius * math.sin(omega * i * dt), 2)]]
        for i in range(n_frames)
    ]

def generate_ctra(radius=5.0, n_frames=NUM_FRAMES, dt=DT, yaw_acc_linear=1.0):
    """CTRA6 모델에 최적화된 등가속 원운동 궤적."""
    alpha = yaw_acc_linear / radius
    return [
        [[round(radius * math.cos(0.5 * alpha * t**2), 2),
          round(radius * math.sin(0.5 * alpha * t**2), 2)]]
        for t in (i * dt for i in range(n_frames))
    ]

def generate_const_acc(a=10.0, v0=0.0, v_const=1.0, n_frames=NUM_FRAMES, dt=DT):
    """
    A: 등가속도 a
    B: 등속도 v_const, y=5.0 고정
    """
    frames = []
    for i in range(n_frames):
        t = i * dt
        x_a = round(0.5 * a * t**2 + v0 * t, 2)
        x_b = round(v_const * t, 2)
        frames.append([[x_a, 0.0], [x_b, 5.0]])
    return frames

def generate_big_fast_circle(radius=20.0, n_frames=NUM_FRAMES, dt=DT):
    """
    A: 고속 원궤적 (1바퀴/전체시간)
    B: 같은 속도로 직선 주행
    """
    T = n_frames * dt
    circumference = 2 * math.pi * radius
    v_line = circumference / T
    frames = []
    for i in range(n_frames):
        t = i * dt
        theta = 2 * math.pi * t / T
        x_arc = round(radius * math.cos(theta), 2)
        y_arc = round(radius * math.sin(theta), 2)
        x_line = round(v_line * t, 2)
        frames.append([[x_arc, y_arc], [x_line, 0.0]])
    return frames

def generate_spiral(r0=5.0, r1=10.0, n_frames=NUM_FRAMES, dt=DT):
    """반지름이 선형 증가하는 나선형 궤적."""
    k = (r1 - r0) / (n_frames * dt)
    omega = 2 * math.pi / (n_frames * dt)
    return [
        [[
            round((r0 + k * t) * math.cos(omega * t), 2),
            round((r0 + k * t) * math.sin(omega * t), 2)
        ]]
        for t in (i * dt for i in range(n_frames))
    ]

# 실제 사용 예시
frames_ctrv_dt          = generate_ctrv()
frames_ctra_dt          = generate_ctra()
frames_const_acc_dt     = generate_const_acc()
frames_big_fast_circle  = generate_big_fast_circle()
frames_spiral           = generate_spiral()
