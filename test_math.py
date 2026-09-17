"""
Test mathematical computations of inter-digital vector spaces (angles, spans, palm normal)
and comprehensive multi-gesture recognition.
"""

import numpy as np
from tracker import MotionCaptureTracker


def test_vector_spaces_and_gestures():
    tracker = MotionCaptureTracker(enable_age=False)

    # 1. Test Vector Spaces Mathematics
    # Build synthetic 21 landmarks representing an open hand facing camera
    synthetic_pts = []
    # Wrist at (0.5, 0.8, 0.0)
    synthetic_pts.append({"id": 0, "x": 0.5, "y": 0.8, "z": 0.0, "px_x": 320, "px_y": 384})

    # Thumb: 1, 2, 3, 4 (spread to the left)
    for i in range(1, 5):
        synthetic_pts.append({"id": i, "x": 0.5 - 0.04 * i, "y": 0.8 - 0.04 * i, "z": 0.0, "px_x": 320 - 25 * i, "px_y": 384 - 20 * i})

    # Index: 5, 6, 7, 8 (pointing up and slightly left)
    for i in range(1, 5):
        synthetic_pts.append({"id": 4 + i, "x": 0.5 - 0.015 * i, "y": 0.7 - 0.05 * i, "z": 0.0, "px_x": 320 - 10 * i, "px_y": 336 - 25 * i})

    # Middle: 9, 10, 11, 12 (pointing straight up)
    for i in range(1, 5):
        synthetic_pts.append({"id": 8 + i, "x": 0.5, "y": 0.7 - 0.055 * i, "z": 0.0, "px_x": 320, "px_y": 336 - 28 * i})

    # Ring: 13, 14, 15, 16 (pointing up and slightly right)
    for i in range(1, 5):
        synthetic_pts.append({"id": 12 + i, "x": 0.5 + 0.015 * i, "y": 0.7 - 0.05 * i, "z": 0.0, "px_x": 320 + 10 * i, "px_y": 336 - 25 * i})

    # Pinky: 17, 18, 19, 20 (spread to the right)
    for i in range(1, 5):
        synthetic_pts.append({"id": 16 + i, "x": 0.5 + 0.035 * i, "y": 0.72 - 0.04 * i, "z": 0.0, "px_x": 320 + 22 * i, "px_y": 345 - 20 * i})

    palm_scale = 0.25
    vs = tracker._compute_vector_spaces(synthetic_pts, palm_scale)

    assert "angles_deg" in vs
    assert "spans_normalized" in vs
    assert "palm_normal" in vs

    angles = vs["angles_deg"]
    assert 0.0 < angles["thumb_index"] < 90.0
    assert 0.0 < angles["index_middle"] < 50.0
    assert 0.0 < angles["middle_ring"] < 50.0
    assert 0.0 < angles["ring_pinky"] < 50.0
    assert angles["total_span"] > angles["index_middle"]

    # 2. Test Gesture Engine
    # Open palm test
    gesture, ext = tracker._classify_gesture(synthetic_pts, pinch_norm=0.5, palm_scale=palm_scale)
    assert gesture in ["OPEN_PALM", "ACTIVE"]

    # Peace sign test: curl ring and pinky toward palm
    peace_pts = list(synthetic_pts)
    # Fold ring and pinky tips close to wrist
    peace_pts[16] = {"id": 16, "x": 0.5, "y": 0.78, "z": 0.0, "px_x": 320, "px_y": 374}
    peace_pts[20] = {"id": 20, "x": 0.5, "y": 0.78, "z": 0.0, "px_x": 320, "px_y": 374}
    peace_pts[4] = {"id": 4, "x": 0.5, "y": 0.75, "z": 0.0, "px_x": 320, "px_y": 360}

    gesture_peace, _ = tracker._classify_gesture(peace_pts, pinch_norm=0.5, palm_scale=palm_scale)
    assert gesture_peace == "PEACE"

    # OK sign test: pinch thumb and index
    gesture_ok, _ = tracker._classify_gesture(synthetic_pts, pinch_norm=0.15, palm_scale=palm_scale)
    assert gesture_ok == "OK_SIGN"

    print("\n[PASSED] Vector spaces and multi-gesture recognition tests succeeded.")


if __name__ == "__main__":
    test_vector_spaces_and_gestures()
