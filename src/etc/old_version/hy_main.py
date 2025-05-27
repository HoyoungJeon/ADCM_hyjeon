import os
import sys
import asyncio
import json
import time
from threading import Thread
from agent import init_agents  # 에이전트 모듈 임포트

# ------------------------------------------------------------------
# Headless 모드: pygame 관련 코드만 제거한 main_4KF.py
# ------------------------------------------------------------------

# 시뮬레이션 및 맵 설정
IS_SIM = True    # True: 시뮬레이션 모드(Headless)로 update_position_MModel 활성화
IS_LOCATION = "merged"  # HDMap 폴더 이름

# HDMap JSON 파일 로딩
src_dir = os.path.dirname(os.path.abspath(__file__))
json_file = os.path.join(src_dir, f"hdmap/{IS_LOCATION}/hdmap.json")
try:
    with open(json_file, "r") as f:
        hdmap_data = json.load(f)
    print(f"HDMAP data loaded: {json_file}")
except (KeyError, FileNotFoundError) as e:
    print(f"Error loading HDMap JSON: {e}", file=sys.stderr)
    sys.exit(1)

# 차량 및 경로 탐색 모듈 임포트
from vehicle import Vehicle, Routing
from hdmap import HDMAP

# HDMap, Routing, Vehicle 객체 생성 (헤드리스 모드)
hdmap = HDMAP.from_json(0, 0, hdmap_data, IS_LOCATION)
routing = Routing(hdmap)
vehicle = Vehicle(hdmap, routing, 0, 0, SIM=IS_SIM)
vehicle.generate_random_route(10)

# 1) Vehicle 상태 업데이트 쓰레드 (60Hz)
def _vehicle_loop():
    while True:
        vehicle.update()
        time.sleep(1/60)
vehicle_thread = Thread(target=_vehicle_loop, daemon=True)
vehicle_thread.start()

# 2) 에이전트 초기화 및 업데이트 쓰레드
agents = init_agents(hdmap)
def _agent_loop(agent):
    while True:
        agent.update(1/60)
        time.sleep(1/60)
for agent in agents:
    t = Thread(target=_agent_loop, args=(agent,), daemon=True)
    t.start()

# ------------------------------------------------------------------
# NATS 클라이언트: 차량 업데이트 구독 & 에이전트 데이터 발행
# ------------------------------------------------------------------
class NatsClient:
    def __init__(self,
                 url: str = "nats://localhost:4222",
                 user: str = "admin",
                 password: str = "password",
                 publish_interval: float = 0.05):
        self.url = url
        self.user = user
        self.password = password
        self.publish_interval = publish_interval
        self.loop = asyncio.new_event_loop()
        self.nc = None
        self.thread = Thread(target=self._start_loop, daemon=True)

    def _start_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._connect_subscribe())
        self.loop.create_task(self._publish_agents())
        self.loop.run_forever()

    async def _connect_subscribe(self):
        from nats.aio.client import Client as NATS
        self.nc = NATS()
        await self.nc.connect(self.url, user=self.user, password=self.password)
        # 차량 상태 토픽 구독
        await self.nc.subscribe("adcm.vehicle.update", cb=self._on_vehicle_update)
        # 장애물 토픽 구독
        await self.nc.subscribe("adcm.vehicle.obstacles", cb=self._on_obstacle)
        print(f"[NATS] Connected and subscribed at {self.url}")

    async def _on_vehicle_update(self, msg):
        data = msg.data.decode()
        vehicle.update_position(json.loads(data))

    async def _on_obstacle(self, msg):
        data = msg.data.decode()
        vehicle.update_obstacle(json.loads(data))

    async def _publish_agents(self):
        while True:
            payload = {
                "agents": [
                    {
                        "id": a.agent_id,
                        "position": a.position,
                        "heading": a.heading,
                        "velocity": a.velocity,
                        "steering_angle": a.steering_angle
                    } for a in agents
                ]
            }
            await self.nc.publish("adcm.sim.obstacles", json.dumps(payload).encode())
            await asyncio.sleep(self.publish_interval)

    def run(self):
        self.thread.start()

    def shutdown(self):
        if self.nc:
            asyncio.run_coroutine_threadsafe(self.nc.close(), self.loop)
        self.loop.call_soon_threadsafe(self.loop.stop)

# NATS 클라이언트 실행
nats_client = NatsClient()
nats_client.run()

# ------------------------------------------------------------------
# 메인 루프 (헤드리스 디버깅): 차량 및 에이전트 정보 출력
# ------------------------------------------------------------------
try:
    while True:
        # Vehicle 정보 출력
        pos = vehicle.position
        vel = vehicle.velocity_kmh
        print(f"Vehicle pos={pos}, speed={vel:.2f} km/h")
        # 각 Agent 정보 출력
        for agent in agents:
            print(f"Agent {agent.agent_id} pos={agent.position}, speed={agent.velocity:.2f} m/s, heading={agent.heading}")
        print("---")
        time.sleep(1)

except KeyboardInterrupt:
    print("Shutting down...")
    vehicle.stop()
    vehicle_thread.join()
    nats_client.shutdown()
    sys.exit(0)