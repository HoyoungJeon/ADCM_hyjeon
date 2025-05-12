import numpy as np


def compute_ttc(p_ego, v_ego, p_tar, v_tar, min_ttc=np.inf):
    """
    p_ego, p_tar: shape (2,) 위치 벡터
    v_ego, v_tar: shape (2,) 속도 벡터
    returns: TTC in seconds (float), or np.inf if no collision
    """
    r_rel = p_tar - p_ego  # 상대 위치
    v_rel = v_tar - v_ego  # 상대 속도
    vr_dot = np.dot(r_rel, v_rel)
    vv_dot = np.dot(v_rel, v_rel)

    # 닫힘 여부 확인
    if vr_dot < 0 and vv_dot > 1e-6:
        ttc = - vr_dot / vv_dot
        # 음수나 너무 작은 값 방지
        return max(ttc, 0.0)
    else:
        return np.inf


# — 예시 사용 —
# ego 상태
p_ego = np.array([5.0, 1.0])
v_ego = np.array([1.2, 0.0])

# target 상태 (KF로부터 추정된 값)
p_tar, v_tar = np.array([6.0, 1.5]), np.array([0.8, -0.2])

ttc = compute_ttc(p_ego, v_ego, p_tar, v_tar)
if np.isfinite(ttc):
    print(f"TTC = {ttc:.2f} s 후 충돌 위험")
else:
    print("충돌 가능성 없음 (TTC = ∞)")
