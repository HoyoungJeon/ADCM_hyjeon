# visualzation.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tracker import MultiObjectTracker
from kalman_filters import IMMEstimator
from frames_data import *
from tracker import predict_future_tracks


def visualize_with_matplotlib():
    # ─── 하드코딩된 입력 데이터 & 설정 ───
    frames = frames_ctrv_dt
    filter_type = 'CTRV6'   # 'CV6', 'CA6', 'CTRV6', 'CTRA6'

    # Tracker 초기화
    tracker = MultiObjectTracker(
        filter_type=filter_type,
        dt=0.1, q_var=0.1, r_var=0.01, # q_var: Process Noise Variance, r_var: Measurement Noise Variance
        max_skipped=3, dist_threshold=10.0, init_frames=3
    )

    # 좌표 범위
    all_pts = [pt for f in frames for pt in f]
    max_x = max((p[0] for p in all_pts), default=0) + 10
    max_y = max((p[1] for p in all_pts), default=0) + 10
    min_x = min((p[0] for p in all_pts), default=0) - 10
    min_y = min((p[1] for p in all_pts), default=0) - 10

    # ─── Matplotlib 셋업 ───
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect('equal', 'box')
    ax.grid(True)

    det_scatter = ax.scatter([], [], c='red',   s=50, label='Detection')
    trk_scatter = ax.scatter([], [], c='green', s=50, label='Track')
    txts = []

    ax.legend(loc='upper left')

    def make_2d(arr):
        """(M,2) 형태 보장. 빈 리스트 → (0,2) 빈 배열."""
        a = np.array(arr, dtype=float)
        if a.size == 0:
            return np.empty((0, 2), dtype=float)
        a = a.reshape(-1, 2)
        return a

    def init():
        det_scatter.set_offsets(np.empty((0, 2)))
        trk_scatter.set_offsets(np.empty((0, 2)))
        return det_scatter, trk_scatter

    def update(frame_idx):
        nonlocal txts
        # 이전 ID 텍스트 삭제
        for t in txts:
            t.remove()
        txts.clear()

        # 트래커에 현재 프레임 적용
        tracker.update(frames[frame_idx])
        track_objs = tracker.tracks
        tracks = tracker.get_tracks()


        # 추가: 과거 히스토리 점 그리기
        for trk in track_objs:
            hist = getattr(trk, 'history', [])
            if hist:
                xs, ys = zip(*hist)
                ax.scatter(xs, ys,
                           s=20,      # 점 크기
                           marker='.',# 마커 모양
                           alpha=0.5)  # 투명도

        # 속력 계산 (vx, vy는 트랙 상태에 포함되어 있다고 가정)
        speeds = [(tr['vx'] ** 2 + tr['vy'] ** 2) ** 0.5 for tr in tracks]

        # 검출점 업데이트
        det_array = make_2d(frames[frame_idx])
        det_scatter.set_offsets(det_array)

        # 트랙 점 업데이트
        trk_pts = [[tr['px'], tr['py']] for tr in tracks]
        trk_array = make_2d(trk_pts)
        trk_scatter.set_offsets(trk_array)

        # 추가: 각 트랙별 미래 5프레임 예측
        future_preds = predict_future_tracks(tracker, steps=5)
        for tid, path in future_preds.items():
            xs, ys = zip(*path)
            # 1) 예측 지점 그리기
            ax.scatter(xs, ys,
                       s=30,  # 점 크기
                       marker='x',  # 마커 모양
                       alpha=0.8,  # 투명도
                       label=f"Pred {tid}" if frame_idx == 0 else None)
            # 2) 점 잇기
            ax.plot(xs, ys,
                    linestyle='--',
                    linewidth=1,
                    alpha=0.6)


        # 속력 텍스트 추가 (ID 아래에 작게)
        for tr, sp in zip(tracks, speeds):
            t = ax.text(
                tr['px'], tr['py'] - 0.2,  # ID 바로 아래 위치
                f"{sp:.1f}",  # 소수점 한 자리 속력
                fontsize=8, color='yellow',
                ha='center', va='top',
                bbox=dict(facecolor='black', alpha=0.5, pad=1)
            )
            txts.append(t)

        # ID 텍스트 추가
        for tr in tracks:
            t = ax.text(
                tr['px'], tr['py'], f"ID:{tr['id']}",
                fontsize=9, color='white',
                bbox=dict(facecolor='black', alpha=0.5, pad=1)
            )
            txts.append(t)

        ax.set_title(f"Frame {frame_idx}")

        # if frame_idx == len(frames) - 1:
        #     plt.close(fig)

        return det_scatter, trk_scatter, *txts

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames),
        init_func=init, interval=1000, blit=False, repeat=False
    )

    plt.show()

if __name__ == '__main__':
    visualize_with_matplotlib()
