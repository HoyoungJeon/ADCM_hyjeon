import copy
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tracker import MultiObjectTracker
from frames_data import *
from tracker import predict_future_tracks


def visualize_with_matplotlib():
    # ─── 입력 데이터 & 설정 ───
    frames = mixed_frames
    filter_type = 'IMM'  # 'CV6', 'CA6', 'VariableTurnEKF', 'FixedTurnEKF', 'IMM'

    # Tracker 초기화
    tracker = MultiObjectTracker(
        filter_type=filter_type,
        dt=0.1, q_var=0.1, r_var=0.01,
        max_skipped=3, dist_threshold=10.0, init_frames=3
    )

    # 좌표 범위 설정
    all_pts = [pt for f in frames for pt in f]
    max_x = max((p[0] for p in all_pts), default=0) + 5
    max_y = max((p[1] for p in all_pts), default=0) + 5
    min_x = min((p[0] for p in all_pts), default=0) - 5
    min_y = min((p[1] for p in all_pts), default=0) - 5

    # ─── Matplotlib 셋업 ───
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect('equal', 'box')
    ax.grid(True)

    def make_2d(arr):
        a = np.array(arr, dtype=float)
        if a.size == 0:
            return np.empty((0, 2), dtype=float)
        return a.reshape(-1, 2)

    def init():
        return []

    def update(frame_idx):
        ax.clear()
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_aspect('equal', 'box')
        ax.grid(True)
        ax.set_title(f"Frame {frame_idx}")

        # 트래커에 현재 프레임 적용
        tracker.update(frames[frame_idx])
        track_objs = tracker.tracks

        # 1) 검출 위치 (빨간 ●)
        det_array = make_2d(frames[frame_idx])
        det_scatter = ax.scatter(
            det_array[:, 0], det_array[:, 1],
            c='red', s=50, marker='o'
        )

        # 2) 궤적 히스토리 및 현재 추정치 그리기
        cmap = plt.get_cmap('tab10')
        lines = [det_scatter]
        labels = ['Detection']

        for idx, trk in enumerate(track_objs):
            color = cmap(idx % 10)
            # 궤적 히스토리
            hist = getattr(trk, 'history', [])
            if len(hist) > 1:
                xs, ys = zip(*hist)
                line, = ax.plot(
                    xs, ys, '-', color=color,
                    linewidth=2, alpha=0.6
                )
                # 속도 계산
                vx, vy = trk.kf.x[2, 0], trk.kf.x[3, 0]
                speed = math.hypot(vx, vy)
                lines.append(line)
                labels.append(f"ID:{trk.track_id}, v={speed:.1f}")

            # 현재 추정치 (초록 ■)
            est = trk.kf.x
            ax.scatter(
                est[0, 0], est[1, 0],
                c=[color], s=60, marker='s'
            )

        # 3) 미래 예측 (검정 점선 + ×)
        future_preds = predict_future_tracks(tracker, steps=50)
        pred_plotted = False
        for tid, path in future_preds.items():
            xs, ys = zip(*path)
            pred_line, = ax.plot(
                xs, ys, '--', marker='x',
                linewidth=1, alpha=0.7, color='black'
            )
            if not pred_plotted:
                lines.append(pred_line)
                labels.append('Prediction')
                pred_plotted = True

        # 4) 범례 (매 프레임 갱신)
        ax.legend(lines, labels, loc='upper left', fontsize='small')

        return []

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames),
        init_func=init, interval=100,
        blit=False, repeat=False
    )

    # paused 상태를 담을 리스트 (closure 를 쉽게 쓰기 위해)
    paused = [False]

    def on_key(event):
        # 아무 키나 누르면 토글
        if paused[0]:
            ani.event_source.start()
        else:
            ani.event_source.stop()
        paused[0] = not paused[0]

    # 키보드 누름 이벤트에 바인딩
    fig.canvas.mpl_connect('key_press_event', on_key)
    # ───────────────────

    plt.show()


if __name__ == '__main__':
    visualize_with_matplotlib()
