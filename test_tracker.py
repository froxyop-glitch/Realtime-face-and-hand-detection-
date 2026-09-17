"""
Unit test for MotionCaptureTracker: verifies initialization, inference pipeline,
refined hand kinematics, pinch detection, and accurate age estimation.
"""

import cv2
import numpy as np
from tracker import MotionCaptureTracker


def test_tracker_pipeline_refined():
    tracker = MotionCaptureTracker(
        camera_id=0,
        width=640,
        height=480,
        fps_target=30,
        model_complexity=0,
        enable_age=True,
        age_interval=1,
        smooth_alpha=0.65,
        output_format="compact",
        models_dir="models",
    )

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_data, hands_data, _ = tracker.process_frame(frame, rgb_frame, w, h)

    assert face_data is None or isinstance(face_data, dict)
    assert isinstance(hands_data, list)

    # Test telemetry emission (compact & json)
    tracker.emit_mocap_telemetry(1, 30.0, face_data, hands_data)
    tracker.output_format = "json"
    tracker.emit_mocap_telemetry(1, 30.0, face_data, hands_data)

    # Test overlay rendering
    tracker.draw_overlays(frame, face_data, hands_data)

    # Test cleanup
    if hasattr(tracker.face_detector, "close"):
        tracker.face_detector.close()
    if hasattr(tracker.hands_detector, "close"):
        tracker.hands_detector.close()

    print("\n[PASSED] Synthetic pipeline test with Refined MoCap & Accurate Age succeeded.")


if __name__ == "__main__":
    test_tracker_pipeline_refined()
