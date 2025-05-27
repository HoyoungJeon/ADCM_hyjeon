import math


frames1 = [
        # Frame 0: A, C
        [[0.0, 0.0], [5.0, 0.0]],
        # Frame 1: A, C
        [[1.0, 0.5], [5.0, 1.0]],
        # Frame 2: A, C
        [[2.0, 1.0], [5.0, 2.0]],
        # Frame 3: A, B, C
        [[3.0, 1.5], [2.0, 5.0], [5.0, 3.0]],
        # Frame 4: A, B, C
        [[4.0, 2.0], [3.0, 5.0], [5.0, 4.0]],
        # Frame 5: A, B, C
        [[5.0, 2.5], [4.0, 5.0], [5.0, 5.0]],
        # Frame 6: A, B
        [[6.0, 3.0], [5.0, 5.0]],
        # Frame 7: A, B
        [[7.0, 3.5], [6.0, 5.0]],
        # Frame 8: A, B
        [[8.0, 4.0], [7.0, 5.0]],
        # Frame 9: A
        [[9.0, 4.5]],
        # Frame 10: A only
        [[10.0, 5.0]],
        # Frame 11: all gone (no detections)
        []
    ]
frames_cross = [
    [[0,0],    [10,0]],        # Frame 0
    [[1,1],    [9, -1]],       # Frame 1
    [[2,2],    [8, -2]],       # Frame 2
    [[3,3],    [7, -3]],       # Frame 3  ← 교차 지점 통과 중
    [[4,4],    [6, -4]],       # Frame 4
    [[5,5],    [5, -5]],       # Frame 5
    [[6,6],    [4, -6]],       # Frame 6
]
frames_entry_exit = [
    [],                                    # Frame 0: 아무도 없음
    [[0,0]],                               # Frame 1: A 등장
    [[1,0]],                               # Frame 2
    [[2,0], [5,5]],                       # Frame 3: B 등장
    [[3,0], [6,5]],                       # Frame 4
    [[4,0], [7,5], [10,10]],             # Frame 5: C 등장
    [[5,0], [8,5], [11,10]],             # Frame 6
    [[6,0],       [9,5]],                # Frame 7: C 퇴장
    [[7,0]],                              # Frame 8: B 퇴장
    [[8,0]],                              # Frame 9
    [],                                    # Frame 10: A 퇴장
]
frames_random = [
    [[0,0], [5,5], [10,0], [15,5]],
    [[0.5,-0.1], [4.8,5.2], [9.7,0.3], [15.2,4.9]],
    [[1.0,0.0], [4.5,5.5], [9.4,0.5], [15.4,4.7]],
    [[1.2,0.3], [4.2,5.8], [9.1,0.7], [15.6,4.5]],
    [[1.5,0.5], [3.9,6.1], [8.8,0.9], [15.8,4.3]],
    [[1.8,0.2], [3.6,5.9], [8.5,1.2], [16.0,4.0]],
    [[2.0,0.0], [3.3,5.6], [8.2,1.0], [16.2,3.8]],
    [[2.3,-0.2], [3.0,5.3], [7.9,0.8], [16.4,3.5]],
]

# 사인 곡선 + 직선 비교용 예제
frames_sine = [
    [
        # A: x 증가 → y=2*sin(0.3*x)
        [round(i * 0.5, 2),
         round(2 * math.sin(0.3 * (i * 0.5)), 2)],
        # B: 일정 속도 0.5 m/frame 로 y=0 → 직선 운동
        [round(i * 0.5, 2), 0.0]
    ]
    for i in range(20)
]



# 공통 파라미터
dt         = 0.1        # 10 Hz 샘플링 주기
num_frames = 20         # 프레임 수
R          = 5.0        # 원 반지름

# 1) 등가속 vs 등속 운동 (CA6 에 최적)
a        = 10.0          # A 객체 가속도 m/s²
v0       = 0.0          # A 객체 초기 속도 m/s
v_const  = 1.0          # B 객체 등속 속도 m/s

# 총 시간 = (num_frames-1) * dt
T_total = (num_frames - 1) * dt





# 1) CTRV6 최적화: 등속 원운동
#    – angular rate ω = 2π / T_total (rad/s)
omega = 2 * math.pi / T_total
frames_ctrv_dt = []
for i in range(num_frames):
    t = i * dt
    θ = omega * t
    x = round(R * math.cos(θ), 2)
    y = round(R * math.sin(θ), 2)
    frames_ctrv_dt.append([[x, y]])

# 2) CTRA6 최적화: 등가속 회전
#    – tangential accel a_t, angular accel α = a_t/R
a_t   = 1.0          # 접선 가속도 (m/s²)
alpha = a_t / R      # rad/s²
frames_ctra_dt = []
for i in range(num_frames):
    t = i * dt
    θ = 0.5 * alpha * t**2      # θ(t)=½αt²
    x = round(R * math.cos(θ), 2)
    y = round(R * math.sin(θ), 2)
    frames_ctra_dt.append([[x, y]])

# dt = 0.1, 10 Hz 샘플링 주기에 맞춘 “곡률 1/R = 0.2”인 반지름 5 m 등속 원운동 예제
frames_ctrv_dt = [
    [[5.0, 0.0]],
    [[4.73, 1.62]],
    [[3.95, 3.07]],
    [[2.73, 4.19]],
    [[1.23, 4.85]],
    [[-0.41, 4.98]],
    [[-2.01, 4.58]],
    [[-3.39, 3.68]],
    [[-4.4, 2.38]],
    [[-4.93, 0.82]],
    [[-4.93, -0.82]],
    [[-4.4, -2.38]],
    [[-3.39, -3.68]],
    [[-2.01, -4.58]],
    [[-0.41, -4.98]],
    [[1.23, -4.85]],
    [[2.73, -4.19]],
    [[3.95, -3.07]],
    [[4.73, -1.62]],
    [[5.0, -0.0]],
]

# dt = 0.1, 접선 가속도 a_t = 1.0 m/s² 로 회전 가속하는 등가속 회전 예제
frames_ctra_dt = [
    [[5.0, 0.0]],
    [[5.0, 0.0]],
    [[5.0, 0.02]],
    [[5.0, 0.04]],
    [[5.0, 0.08]],
    [[5.0, 0.12]],
    [[5.0, 0.18]],
    [[4.99, 0.24]],
    [[4.99, 0.32]],
    [[4.98, 0.4]],
    [[4.98, 0.5]],
    [[4.96, 0.6]],
    [[4.95, 0.72]],
    [[4.93, 0.84]],
    [[4.9, 0.97]],
    [[4.87, 1.12]],
    [[4.84, 1.27]],
    [[4.79, 1.42]],
    [[4.74, 1.59]],
    [[4.68, 1.77]],
]



frames_const_acc_dt = []
for i in range(num_frames):
    t = i * dt
    # A: 위치 = 0.5 * a * t² + v0 * t
    x_a = 0.5 * a * t**2 + v0 * t
    y_a = 0.0
    # B: 위치 = v_const * t
    x_b = v_const * t
    y_b = 5.0

    frames_const_acc_dt.append([
        [round(x_a, 2), round(y_a, 2)],
        [round(x_b, 2), round(y_b, 2)]
    ])

# 2) 대형 고속 원 궤적 vs 직선 (CTRV/CTRA에 최적)
r       = 20.0                       # 반지름 20 m
T_total = num_frames * dt           # 전체 주기 (s)
circumf = 2 * math.pi * r

frames_big_fast_circle_dt = []
for i in range(num_frames):
    t = i * dt
    # A: 고속 원 궤적
    theta = 2 * math.pi * (t / T_total)  # 초당 한 바퀴
    x_arc = r * math.cos(theta)
    y_arc = r * math.sin(theta)
    # B: 같은 속도로 직선 주행 (circumf/T_total m/s)
    v_line = circumf / T_total
    x_line = v_line * t
    y_line = 0.0

    frames_big_fast_circle_dt.append([
        [round(x_arc, 2), round(y_arc, 2)],
        [round(x_line, 2), round(y_line, 2)]
    ])