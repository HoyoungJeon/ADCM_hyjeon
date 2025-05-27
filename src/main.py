import numpy as np
from scipy.optimize import linear_sum_assignment
from tracker import MultiObjectTracker, predict_future_tracks
from frames_data import *

if __name__ == '__main__':
    filter_type = 'IMM'  # 'CV6', 'CA6', 'CTRV6', 'CTRA6', 'IMM'
    tracker = MultiObjectTracker(
        filter_type=filter_type,
        dt=0.1, q_var=0.1, r_var=0.01, # q_var: Process Noise Variance, r_var: Measurement Noise Variance
        max_skipped=3, dist_threshold=10, init_frames=3
    )
    frames = frames_spiral

    model_names = ['CV6', 'CA6', 'CTRV6', 'CTRA6']

    for idx, dets in enumerate(frames):
        print(f"\n--- Frame {idx} ---")
        print("Detections:", dets)

        tracker.update(dets)

        # 실제 Track 객체 리스트와, get_tracks()로 만든 dict 리스트를 병렬로 가져옵니다.
        track_objs = tracker.tracks
        tracks     = tracker.get_tracks()

        if not tracks:
            print("No confirmed tracks yet.")
        else:
            for obj, tr in zip(track_objs, tracks):
                # 기본 상태 출력
                line = (
                    f"Track {tr['id']}: px={tr['px']:.1f}, py={tr['py']:.1f}, "
                    f"vx={tr['vx']:.1f}, vy={tr['vy']:.1f}, "
                    f"ax={tr['ax']:.1f}, ay={tr['ay']:.1f}. "
                    f"Skipped frame:{tr['skipped_frames']}"
                )
                # IMM이면 모든 서브모델 확률 출력
                if filter_type == 'IMM':
                    probs = obj.kf.mu
                    prob_strs = [f"{name}: {p*100:.1f}%"
                                 for name, p in zip(model_names, probs)]
                    line += "  [" + ", ".join(prob_strs) + "]"
                print(line)

            future_preds = predict_future_tracks(tracker, steps=5)
            for tid, path in future_preds.items():
                print(f"\n--- Track {tid} 5-frame Prediction ---")
                for i, (px, py) in enumerate(path, start=1):
                    print(f" +{i}: px={px:.2f}, py={py:.2f}")


