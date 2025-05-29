import numpy as np
from tracker import MultiObjectTracker, predict_future_tracks
from frames_data import *

if __name__ == '__main__':
    # Instantiate the tracker (IMM only)
    tracker = MultiObjectTracker(
        dt=0.1,         # time step
        q_var=0.1,      # process noise variance
        r_var=0.01,     # measurement noise variance
        max_skipped=3,  # max allowed consecutive misses
        dist_threshold=10,
        init_frames=3   # detections needed to confirm a track
    )

    frames = mixed_frames
    model_names = ['CV6', 'CA6', 'VariableTurnEKF', 'FixedTurnEKF']

    for idx, detections in enumerate(frames):
        print(f"\n--- Frame {idx} ---")
        print("Detections:", detections)

        tracker.update(detections)

        # Parallel lists: the raw Track objects, and the dicts from get_tracks()
        track_objs = tracker.tracks
        tracks_info = tracker.get_tracks()

        if not tracks_info:
            print("No confirmed tracks yet.")
        else:
            for obj, info in zip(track_objs, tracks_info):
                # Base state output
                line = (
                    f"Track {info['id']}: "
                    f"px={info['px']:.1f}, py={info['py']:.1f}, "
                    f"vx={info['vx']:.1f}, vy={info['vy']:.1f}, "
                    f"ax={info['ax']:.1f}, ay={info['ay']:.1f}. "
                    f"Skipped frames: {info['skipped_frames']}"
                )
                # Print IMM sub-model probabilities
                probs = obj.kf.mu
                prob_strs = [f"{name}: {p*100:.1f}%" for name, p in zip(model_names, probs)]
                line += "  [" + ", ".join(prob_strs) + "]"
                print(line)

            future_preds = predict_future_tracks(tracker, steps=5)
            # To print 5-step ahead predictions, uncomment below:
            # for track_id, path in future_preds.items():
            #     print(f"\n--- Track {track_id} 5-frame Prediction ---")
            #     for step, (px, py) in enumerate(path, start=1):
            #         print(f"+{step}: px={px:.2f}, py={py:.2f}")
