# Visualzation.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tracker import MultiObjectTracker

def visualize_with_matplotlib():
    # ─── 하드코딩된 입력 데이터 & 설정 ───
    frames = [
        [[0.0, 0.0], [5.0, 0.0]],
        [[1.0, 0.5], [5.0, 1.0]],
        [[2.0, 1.0], [5.0, 2.0]],
        [[3.0, 1.5], [2.0, 5.0], [5.0, 3.0]],
        [[4.0, 2.0], [3.0, 5.0], [5.0, 4.0]],
        [[5.0, 2.5], [4.0, 5.0], [5.0, 5.0]],
        [[6.0, 3.0], [5.0, 5.0]],
        [[7.0, 3.5], [6.0, 5.0]],
        [[8.0, 4.0], [7.0, 5.0]],
        [[9.0, 4.5]],
        [[10.0, 5.0]],
        []
    ]
    filter_type = 'CTRV6'   # 'CV6', 'CA6', 'CTRV6', 'CTRA6'

    # Tracker 초기화
    tracker = MultiObjectTracker(
        filter_type=filter_type,
        dt=1.0, q_var=1.0, r_var=1.0,
        max_skipped=3, dist_threshold=10.0, init_frames=3
    )

    # 좌표 범위
    all_pts = [pt for f in frames for pt in f]
    max_x = max((p[0] for p in all_pts), default=0) + 10
    max_y = max((p[1] for p in all_pts), default=0) + 10

    # ─── Matplotlib 셋업 ───
    fig, ax = plt.subplots()
    ax.set_xlim(-3, max_x)
    ax.set_ylim(-3, max_y)
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
        tracks = tracker.get_tracks()

        # 검출점 업데이트
        det_array = make_2d(frames[frame_idx])
        det_scatter.set_offsets(det_array)

        # 트랙 점 업데이트
        trk_pts = [[tr['px'], tr['py']] for tr in tracks]
        trk_array = make_2d(trk_pts)
        trk_scatter.set_offsets(trk_array)

        # ID 텍스트 추가
        for tr in tracks:
            t = ax.text(
                tr['px'], tr['py'], f"ID:{tr['id']}",
                fontsize=9, color='white',
                bbox=dict(facecolor='black', alpha=0.5, pad=1)
            )
            txts.append(t)

        ax.set_title(f"Frame {frame_idx}")
        return det_scatter, trk_scatter, *txts

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames),
        init_func=init, interval=1000, blit=False
    )

    plt.show()

if __name__ == '__main__':
    visualize_with_matplotlib()
