import numpy as np
from scipy.optimize import linear_sum_assignment
from tracker import MultiObjectTracker

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




if __name__ == '__main__':
    filter_type = 'CTRV6'  # 'CV6', 'CA6', 'CTRV6', 'CTRA6'
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