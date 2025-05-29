import math
import numpy as np
from kalman_filters import KalmanFilterCV6, KalmanFilterCA6, VariableTurnEKF, FixedTurnEKF

# ── 공통 파라미터 ──
DT         = 0.1        # 프레임 간 시간 간격 (초)
NUM_FRAMES = 20         # 프레임 수

# ── 함수 기반 생성 시퀀스 ──
def generate_sine_vs_line(n_frames=NUM_FRAMES, x_step=0.5, amplitude=2, freq=0.3):
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

# ── 물리 모델 기반 생성 시퀀스 ──
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

# ── 에이전트 기반 생성 시퀀스 ──
class Agent:
    def __init__(self, model_cls, init_state, dt=DT, q_var=0.0, r_var=0.0):
        self.kf = model_cls(dt=dt, q_var=q_var, r_var=r_var)
        for i, v in enumerate(init_state):
            self.kf.x[i, 0] = v
        self.history = []

    def step(self):
        state = self.kf.predict()
        pos = (float(state[0, 0]), float(state[1, 0]))
        self.history.append(pos)
        return pos


def generate_from_agents(agent_configs, num_frames=NUM_FRAMES):
    agents = []
    for cfg in agent_configs:
        if isinstance(cfg, dict):
            agents.append(Agent(cfg['model'], cfg['init']))
        else:
            # 이미 step() 메서드를 가진 에이전트 인스턴스로 간주
            agents.append(cfg)
    frames = []
    for _ in range(num_frames):
        detections = [agent.step() for agent in agents]
        frames.append(detections)
    return frames

class RandomWaypointAgent:
    def __init__(self, bounds, speed, dt):
        self.bounds = bounds
        self.speed = speed
        self.dt = dt
        self.pos = np.random.uniform([bounds[0], bounds[2]], [bounds[1], bounds[3]])
        self._pick_new_waypoint()

    def _pick_new_waypoint(self):
        x0, x1, y0, y1 = self.bounds
        self.wpt = np.random.uniform([x0, y0], [x1, y1])

    def step(self):
        dir_vec = self.wpt - self.pos
        dist = np.linalg.norm(dir_vec)
        if dist < self.speed * self.dt:
            self.pos = self.wpt.copy()
            self._pick_new_waypoint()
        else:
            self.pos += dir_vec / dist * self.speed * self.dt
        return tuple(self.pos)

class RandomWalkAgent:
    def __init__(self, init_pos, init_vel, dt, max_accel, max_speed):
        self.pos = np.array(init_pos, float)
        self.vel = np.array(init_vel, float)
        self.dt = dt
        self.max_accel = max_accel
        self.max_speed = max_speed

    def step(self):
        accel = np.random.uniform(-self.max_accel, self.max_accel, size=2)
        self.vel += accel * self.dt
        speed = np.linalg.norm(self.vel)
        if speed > self.max_speed:
            self.vel = self.vel / speed * self.max_speed
        self.pos += self.vel * self.dt
        return tuple(self.pos)

class CircularMotionAgent:
    def __init__(self, center, radius, omega, dt):
        self.cx, self.cy = center
        self.R = radius
        self.omega = omega
        self.dt = dt
        self.angle = 0.0

    def step(self):
        self.angle += self.omega * self.dt
        x = self.cx + self.R * np.cos(self.angle)
        y = self.cy + self.R * np.sin(self.angle)
        return (x, y)

# ── 실제 사용 예시: 여러 에이전트를 섞어 프레임 생성 ──
kf_agent = Agent(KalmanFilterCV6, init_state=[0, 0, 1, 0, 0, 0])
rw_agent = RandomWalkAgent(init_pos=(5, 5), init_vel=(0, 0), dt=DT, max_accel=1.0, max_speed=3.0)
wp_agent = RandomWaypointAgent(bounds=[-10, 10, -10, 10], speed=2.0, dt=DT)
# 곡선 주행 에이전트 추가
circ_agent = CircularMotionAgent(center=(0, 0), radius=10.0, omega=0.2, dt=DT)
frames_big_fast_circle  = generate_ctrv()
mixed_frames = generate_from_agents([
    {'model': KalmanFilterCV6, 'init': [0, 0, 1, 0, 0, 0]},
    {'model': KalmanFilterCA6, 'init': [5, 5, 0, 0, 0.2, -0.1]},
    {'model': VariableTurnEKF, 'init': [10, 0, 2, math.pi/4, 0.1, 0.05]},
    circ_agent,
    rw_agent,
    wp_agent
], num_frames=100)
