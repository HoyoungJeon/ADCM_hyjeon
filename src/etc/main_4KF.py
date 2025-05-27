import numpy as np
from scipy.optimize import linear_sum_assignment
from tracker import MultiObjectTracker
from frames_data import *




if __name__ == '__main__':
    filter_type = 'IMM'  # 'CV6', 'CA6', 'CTRV6', 'CTRA6', 'IMM'
    tracker = MultiObjectTracker(
        filter_type=filter_type,
        dt=1.0, q_var=1.0, r_var=1.0,
        max_skipped=3, dist_threshold=10, init_frames=3
    )
    frames = frames_cross
    for idx, dets in enumerate(frames):
        print(f"\n--- Frame {idx} ---")
        print("Detections:", dets)
        tracker.update(dets)
        tracks = tracker.get_tracks()
        if not tracks:
            print("No confirmed tracks yet.")
        else:
            for tr in tracks:
                print(
                    f"Track {tr['id']}: px={tr['px']:.1f}, py={tr['py']:.1f}, "
                    f"vx={tr['vx']:.1f}, vy={tr['vy']:.1f}, "
                    f"ax={tr['ax']:.1f}, ay={tr['ay']:.1f}. "
                    f"Skipped frame:{tr['skipped_frames']}"
                )