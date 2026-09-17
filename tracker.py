# ==============================================================================
# Installation Requirements:
# Run the following command in your terminal before running this script:
#   pip install opencv-python mediapipe numpy
#
# (Note: Use standard 'opencv-python' rather than 'opencv-python-headless'
#  so that cv2.imshow GUI windows can be rendered)
# ==============================================================================

import argparse
import json
import math
import os
import sys
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


class MotionCaptureTracker:
    """
    High-performance real-time Face, Accurate Age, and Multi-Gesture Hand MoCap Tracker.
    Features:
      - Inter-digital vector spaces: 3D direction unit vectors, inter-finger angles (θ),
        fingertip span distances, and 3D palm normal plane orientation.
      - Comprehensive gesture engine: OPEN_PALM, FIST, PEACE, OK_SIGN, ROCK_ON, SPIDERMAN,
        CALL_ME, THUMBS_UP, THUMBS_DOWN, POINTING, GUN, THREE, FOUR, PINCH.
      - Temporal Exponential Moving Average (EMA) landmark anti-jitter smoothing.
      - Aspect-ratio preserving square face cropping with softmax probability smoothing.
      - Cyber-HUD with vector space webbing, finger gradients, and MoCap telemetry stream.
    """

    # Official model download URLs
    FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
    HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    AGE_MODEL_URL = "https://huggingface.co/onnxmodelzoo/age_googlenet/resolve/main/age_googlenet.onnx"

    AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
    AGE_MIDPOINTS = np.array([1.5, 5.0, 10.0, 17.5, 28.5, 40.5, 50.5, 70.0], dtype=np.float32)

    # Distinct neon colors for refined skeletal finger tracking (BGR format)
    FINGER_COLORS = {
        "thumb": (255, 220, 0),       # Cyan / Light Blue
        "index": (0, 255, 128),       # Neon Spring Green
        "middle": (0, 255, 255),      # Neon Yellow
        "ring": (0, 165, 255),        # Neon Orange
        "pinky": (255, 0, 230),       # Neon Magenta
        "palm": (200, 200, 200),      # Clean Gray/White
    }

    # Joint connection chains for individual fingers
    FINGER_CHAINS = {
        "thumb": [(0, 1), (1, 2), (2, 3), (3, 4)],
        "index": [(0, 5), (5, 6), (6, 7), (7, 8)],
        "middle": [(9, 10), (10, 11), (11, 12)],
        "ring": [(13, 14), (14, 15), (15, 16)],
        "pinky": [(0, 17), (17, 18), (18, 19), (19, 20)],
        "palm": [(5, 9), (9, 13), (13, 17)],
    }

    # Visual symbols for recognized gestures
    GESTURE_ICONS = {
        "OPEN_PALM": "OPEN",
        "FIST": "FIST",
        "PEACE": "PEACE",
        "OK_SIGN": "OK",
        "ROCK_ON": "ROCK",
        "SPIDERMAN": "ILY",
        "CALL_ME": "SHAKA",
        "THUMBS_UP": "THUMB-UP",
        "THUMBS_DOWN": "THUMB-DN",
        "POINTING": "POINT",
        "GUN": "GUN",
        "THREE": "THREE",
        "FOUR": "FOUR",
        "PINCH": "PINCH",
        "ACTIVE": "ACTIVE",
    }

    def __init__(
        self,
        camera_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps_target: int = 30,
        model_complexity: int = 0,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        enable_age: bool = True,
        age_interval: int = 4,
        smooth_alpha: float = 0.65,
        output_format: str = "compact",
        models_dir: str = "models",
    ):
        """
        Initialize the tracker and configure MediaPipe and Age DNN modules for low latency.

        :param camera_id: Device index for webcam (default: 0).
        :param width: Frame width resolution (default: 640 for low latency).
        :param height: Frame height resolution (default: 480 for low latency).
        :param fps_target: Target capture frame rate.
        :param model_complexity: 0 for ultra-fast/low-latency, 1 for high precision.
        :param min_detection_confidence: Minimum detection confidence threshold.
        :param min_tracking_confidence: Minimum tracking confidence threshold.
        :param enable_age: Whether to enable real-time age classification.
        :param age_interval: Run age inference every N frames to maintain maximum FPS.
        :param smooth_alpha: EMA smoothing coefficient for hand landmarks (0.1=heavy smooth, 1.0=raw).
        :param output_format: 'compact' (live summary line) or 'json' (complete serialized landmark data).
        :param models_dir: Directory to cache model assets.
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps_target = fps_target
        self.enable_age = enable_age
        self.age_interval = age_interval
        self.smooth_alpha = smooth_alpha
        self.output_format = output_format
        self.models_dir = models_dir

        # Enable OpenCV internal multithreading & optimizations
        cv2.setUseOptimized(True)

        # Performance / FPS state
        self.prev_time = 0.0
        self.fps = 0.0
        self.frame_count = 0

        # State for temporal smoothing
        self.smoothed_landmarks: Dict[str, List[Dict[str, float]]] = {}
        self.prev_wrist_positions: Dict[str, Tuple[float, float, float, float]] = {}
        self.age_probs_smoothed: Optional[np.ndarray] = None
        self.cached_age_data: Optional[Dict[str, Any]] = None

        # Check whether legacy solutions API or modern Tasks API is active
        self.use_legacy_solutions = hasattr(mp, "solutions")

        if self.use_legacy_solutions:
            self._init_legacy_solutions(
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        else:
            self._init_tasks_api(
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )

        # Initialize ONNX Age Detector
        if self.enable_age:
            self._init_age_detector()

    # --------------------------------------------------------------------------
    # Initializers for Face, Hand, and Age Modules
    # --------------------------------------------------------------------------
    def _init_age_detector(self):
        """Initialize ONNX-based Age Detector via OpenCV DNN module."""
        os.makedirs(self.models_dir, exist_ok=True)
        age_model_path = os.path.join(self.models_dir, "age_googlenet.onnx")

        if not os.path.exists(age_model_path):
            print(f"[Init] Downloading ONNX Age Detector model to {age_model_path}...")
            req = urllib.request.Request(self.AGE_MODEL_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp, open(age_model_path, "wb") as f:
                f.write(resp.read())

        print("[Init] Loading ONNX Age Detector into OpenCV DNN...")
        self.age_net = cv2.dnn.readNetFromONNX(age_model_path)
        self.age_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

    def _init_legacy_solutions(
        self,
        model_complexity: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
    ):
        """Initialize legacy MediaPipe Solutions modules (mediapipe < 0.10.15)."""
        print("[Init] Using MediaPipe Solutions API backend.")
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands

        self.face_detector = self.mp_face.FaceDetection(
            min_detection_confidence=min_detection_confidence,
            model_selection=0,
        )

        self.hands_detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def _init_tasks_api(
        self,
        min_detection_confidence: float,
        min_tracking_confidence: float,
    ):
        """Initialize modern MediaPipe Tasks Vision modules (mediapipe >= 0.10.15 / 1.0+)."""
        print("[Init] Using modern MediaPipe Tasks API backend (TensorFlow Lite XNNPACK).")
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision

        self.vision = vision
        self.drawing_utils = vision.drawing_utils
        self.drawing_styles = vision.drawing_styles
        self.hand_connections = vision.HandLandmarksConnections.HAND_CONNECTIONS

        os.makedirs(self.models_dir, exist_ok=True)
        face_model_path = os.path.join(self.models_dir, "blaze_face_short_range.tflite")
        hand_model_path = os.path.join(self.models_dir, "hand_landmarker.task")

        if not os.path.exists(face_model_path):
            print(f"[Init] Downloading face detector model to {face_model_path}...")
            urllib.request.urlretrieve(self.FACE_MODEL_URL, face_model_path)

        if not os.path.exists(hand_model_path):
            print(f"[Init] Downloading hand landmarker model to {hand_model_path}...")
            urllib.request.urlretrieve(self.HAND_MODEL_URL, hand_model_path)

        face_options = vision.FaceDetectorOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=face_model_path),
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=min_detection_confidence,
        )
        self.face_detector = vision.FaceDetector.create_from_options(face_options)

        hand_options = vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=hand_model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.hands_detector = vision.HandLandmarker.create_from_options(hand_options)

    # --------------------------------------------------------------------------
    # Camera Initialization
    # --------------------------------------------------------------------------
    def init_camera(self) -> cv2.VideoCapture:
        """Initialize camera stream with low latency settings."""
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(self.camera_id)
        else:
            cap = cv2.VideoCapture(self.camera_id)

        if not cap.isOpened():
            raise RuntimeError(
                f"Error: Could not open camera at index {self.camera_id}. "
                "Ensure your camera is connected and not occupied by another app."
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps_target)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def calculate_fps(self) -> float:
        """Compute real-time processing FPS using Exponential Moving Average."""
        curr_time = time.perf_counter()
        if self.prev_time > 0:
            time_diff = curr_time - self.prev_time
            if time_diff > 0:
                instant_fps = 1.0 / time_diff
                self.fps = 0.85 * self.fps + 0.15 * instant_fps
        self.prev_time = curr_time
        return self.fps

    # --------------------------------------------------------------------------
    # Refined Age Classification (Square Crop + Softmax EMA Ensemble)
    # --------------------------------------------------------------------------
    def predict_age(self, bgr_frame: np.ndarray, bbox_px: Dict[str, int]) -> Optional[Dict[str, Any]]:
        """
        High-accuracy age prediction with aspect-ratio preserving square crop
        and rolling temporal probability smoothing.
        """
        if not self.enable_age or not hasattr(self, "age_net"):
            return None

        h, w, _ = bgr_frame.shape
        xmin = bbox_px["xmin"]
        ymin = bbox_px["ymin"]
        bw = bbox_px["width"]
        bh = bbox_px["height"]

        if bw <= 10 or bh <= 10:
            return None

        # Aspect-ratio preserving square crop with 35% forehead/chin context
        cx = xmin + (bw / 2.0)
        cy = ymin + (bh / 2.0)
        side = int(max(bw, bh) * 1.35)

        x1 = int(cx - (side / 2.0))
        y1 = int(cy - (side / 2.0))
        x2 = x1 + side
        y2 = y1 + side

        pad_top = max(0, -y1)
        pad_bottom = max(0, y2 - h)
        pad_left = max(0, -x1)
        pad_right = max(0, x2 - w)

        crop_y1 = max(0, y1)
        crop_y2 = min(h, y2)
        crop_x1 = max(0, x1)
        crop_x2 = min(w, x2)

        face_crop = bgr_frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if face_crop.size == 0 or face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            return None

        # Replicate borders if face touches boundary to preserve square geometry
        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            face_crop = cv2.copyMakeBorder(
                face_crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE
            )

        # Forward pass through DNN
        blob = cv2.dnn.blobFromImage(
            face_crop,
            scalefactor=1.0,
            size=(224, 224),
            mean=(104.0, 117.0, 123.0),
            swapRB=False,
            crop=False,
        )
        self.age_net.setInput(blob)
        raw_preds = self.age_net.forward()[0]

        probs = np.array(raw_preds, dtype=np.float32)
        if np.sum(probs) > 0:
            probs = probs / np.sum(probs)

        # Temporal EMA smoothing
        beta = 0.28
        if self.age_probs_smoothed is None:
            self.age_probs_smoothed = probs
        else:
            self.age_probs_smoothed = beta * probs + (1.0 - beta) * self.age_probs_smoothed

        expected_age = float(np.sum(self.age_probs_smoothed * self.AGE_MIDPOINTS))
        best_idx = int(np.argmax(self.age_probs_smoothed))
        confidence = float(self.age_probs_smoothed[best_idx])
        bracket = self.AGE_BUCKETS[best_idx]

        return {
            "expected_age": round(expected_age, 1),
            "bracket": bracket,
            "confidence": round(confidence, 2),
            "distribution": [round(float(p), 3) for p in self.age_probs_smoothed],
        }

    # --------------------------------------------------------------------------
    # Inter-Digital Vector Space & Geometric Calculations
    # --------------------------------------------------------------------------
    @staticmethod
    def _compute_vector_spaces(pts: List[Dict[str, Any]], palm_scale: float) -> Dict[str, Any]:
        """
        Compute inter-digital vector spaces:
          - 3D unit vectors along each finger
          - Angles between adjacent finger vector spaces (in degrees)
          - Normalized inter-fingertip spans
          - 3D Palm Plane Normal Vector
        """
        # Finger tip IDs: Thumb=4, Index=8, Middle=12, Ring=16, Pinky=20
        # Finger base MCP IDs: Thumb=2, Index=5, Middle=9, Ring=13, Pinky=17
        mcp_ids = [2, 5, 9, 13, 17]
        tip_ids = [4, 8, 12, 16, 20]

        # 1. Compute 3D Direction Vectors for each finger (Base -> Tip)
        finger_vectors = []
        for base_id, tip_id in zip(mcp_ids, tip_ids):
            vec = np.array([
                pts[tip_id]["x"] - pts[base_id]["x"],
                pts[tip_id]["y"] - pts[base_id]["y"],
                pts[tip_id]["z"] - pts[base_id]["z"],
            ], dtype=np.float32)
            finger_vectors.append(vec)

        # Helper: angle between two 3D vectors in degrees
        def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 < 1e-6 or norm2 < 1e-6:
                return 0.0
            cos_theta = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
            return float(np.degrees(np.arccos(cos_theta)))

        # 2. Inter-finger Vector Space Angles
        theta_thumb_index = angle_between(finger_vectors[0], finger_vectors[1])
        theta_index_middle = angle_between(finger_vectors[1], finger_vectors[2])
        theta_middle_ring = angle_between(finger_vectors[2], finger_vectors[3])
        theta_ring_pinky = angle_between(finger_vectors[3], finger_vectors[4])
        theta_total_span = angle_between(finger_vectors[0], finger_vectors[4])

        # 3. Inter-fingertip Span Distances (Normalized by Palm Scale)
        def tip_dist(id1: int, id2: int) -> float:
            d = math.sqrt(
                (pts[id1]["x"] - pts[id2]["x"]) ** 2
                + (pts[id1]["y"] - pts[id2]["y"]) ** 2
                + (pts[id1]["z"] - pts[id2]["z"]) ** 2
            )
            return d / palm_scale

        span_ti = tip_dist(4, 8)
        span_im = tip_dist(8, 12)
        span_mr = tip_dist(12, 16)
        span_rp = tip_dist(16, 20)
        span_total = tip_dist(4, 20)

        # 4. Palm Normal Vector via Cross Product of Palm Triangle (Wrist -> Index_MCP x Wrist -> Pinky_MCP)
        wrist = np.array([pts[0]["x"], pts[0]["y"], pts[0]["z"]], dtype=np.float32)
        index_mcp = np.array([pts[5]["x"], pts[5]["y"], pts[5]["z"]], dtype=np.float32)
        pinky_mcp = np.array([pts[17]["x"], pts[17]["y"], pts[17]["z"]], dtype=np.float32)

        v_u = index_mcp - wrist
        v_v = pinky_mcp - wrist
        palm_cross = np.cross(v_u, v_v)
        norm_cross = np.linalg.norm(palm_cross)
        palm_normal = (palm_cross / norm_cross).tolist() if norm_cross > 1e-6 else [0.0, 0.0, 1.0]

        return {
            "angles_deg": {
                "thumb_index": round(theta_thumb_index, 1),
                "index_middle": round(theta_index_middle, 1),
                "middle_ring": round(theta_middle_ring, 1),
                "ring_pinky": round(theta_ring_pinky, 1),
                "total_span": round(theta_total_span, 1),
            },
            "spans_normalized": {
                "thumb_index": round(span_ti, 2),
                "index_middle": round(span_im, 2),
                "middle_ring": round(span_mr, 2),
                "ring_pinky": round(span_rp, 2),
                "total_span": round(span_total, 2),
            },
            "palm_normal": [round(float(n), 3) for n in palm_normal],
        }

    # --------------------------------------------------------------------------
    # Comprehensive Multi-Gesture Recognition Engine
    # --------------------------------------------------------------------------
    @staticmethod
    def _classify_gesture(
        smoothed_pts: List[Dict[str, Any]],
        pinch_norm: float,
        palm_scale: float,
    ) -> Tuple[str, Dict[str, bool]]:
        """
        Classify from an extensive library of hand gestures:
          - OPEN_PALM, FIST, PEACE, OK_SIGN, ROCK_ON, SPIDERMAN, CALL_ME,
            THUMBS_UP, THUMBS_DOWN, POINTING, GUN, THREE, FOUR, PINCH.
        """
        wrist = smoothed_pts[0]

        # Finger extension check: Tip farther from Wrist than PIP
        def is_ext(tip_id: int, pip_id: int) -> bool:
            d_tip = (smoothed_pts[tip_id]["x"] - wrist["x"]) ** 2 + (smoothed_pts[tip_id]["y"] - wrist["y"]) ** 2
            d_pip = (smoothed_pts[pip_id]["x"] - wrist["x"]) ** 2 + (smoothed_pts[pip_id]["y"] - wrist["y"]) ** 2
            return d_tip > d_pip * 1.08

        # Thumb extension relative to Pinky MCP
        d_thumb_tip = (smoothed_pts[4]["x"] - smoothed_pts[17]["x"]) ** 2 + (smoothed_pts[4]["y"] - smoothed_pts[17]["y"]) ** 2
        d_thumb_mcp = (smoothed_pts[2]["x"] - smoothed_pts[17]["x"]) ** 2 + (smoothed_pts[2]["y"] - smoothed_pts[17]["y"]) ** 2
        thumb_ext = d_thumb_tip > d_thumb_mcp * 1.15

        index_ext = is_ext(8, 6)
        middle_ext = is_ext(12, 10)
        ring_ext = is_ext(16, 14)
        pinky_ext = is_ext(20, 18)

        fingers_ext = {
            "thumb": thumb_ext,
            "index": index_ext,
            "middle": middle_ext,
            "ring": ring_ext,
            "pinky": pinky_ext,
        }
        ext_count = sum(fingers_ext.values())

        # Vertical orientation of thumb (y points downward in image coordinates)
        thumb_dy = smoothed_pts[4]["y"] - smoothed_pts[2]["y"]

        # 1. OK Sign: Thumb and Index touching, other 3 extended
        if pinch_norm < 0.26 and middle_ext and ring_ext and pinky_ext:
            return "OK_SIGN", fingers_ext

        # 2. Pinch: Thumb and Index touching, other fingers not all extended
        if pinch_norm < 0.26 and (not middle_ext or not ring_ext):
            return "PINCH", fingers_ext

        # 3. Peace / Victory: Index and Middle extended, Ring and Pinky curled
        if index_ext and middle_ext and not ring_ext and not pinky_ext:
            return "PEACE", fingers_ext

        # 4. Spiderman (ILY sign): Thumb, Index, Pinky extended; Middle and Ring curled
        if thumb_ext and index_ext and pinky_ext and not middle_ext and not ring_ext:
            return "SPIDERMAN", fingers_ext

        # 5. Rock On / Horns: Index and Pinky extended, Middle and Ring curled (Thumb folded)
        if index_ext and pinky_ext and not middle_ext and not ring_ext:
            return "ROCK_ON", fingers_ext

        # 6. Call Me / Shaka: Thumb and Pinky extended, Index, Middle, Ring curled
        if thumb_ext and pinky_ext and not index_ext and not middle_ext and not ring_ext:
            return "CALL_ME", fingers_ext

        # 7. Thumbs Up / Thumbs Down: Thumb extended, all 4 other fingers curled
        if thumb_ext and not index_ext and not middle_ext and not ring_ext and not pinky_ext:
            if thumb_dy < -0.06:
                return "THUMBS_UP", fingers_ext
            elif thumb_dy > 0.06:
                return "THUMBS_DOWN", fingers_ext

        # 8. Gun: Thumb up, Index extended forward, others curled
        if thumb_ext and index_ext and not middle_ext and not ring_ext and not pinky_ext:
            return "GUN", fingers_ext

        # 9. Pointing: Index finger extended, others curled
        if index_ext and not middle_ext and not ring_ext and not pinky_ext and not thumb_ext:
            return "POINTING", fingers_ext

        # 10. Four Fingers: 4 fingers extended, thumb curled
        if not thumb_ext and index_ext and middle_ext and ring_ext and pinky_ext:
            return "FOUR", fingers_ext

        # 11. Three Fingers: 3 fingers extended
        if ext_count == 3:
            return "THREE", fingers_ext

        # 12. Open Palm: All or almost all fingers extended
        if ext_count >= 4:
            return "OPEN_PALM", fingers_ext

        # 13. Fist: All fingers curled
        if ext_count == 0:
            return "FIST", fingers_ext

        return "ACTIVE", fingers_ext

    # --------------------------------------------------------------------------
    # Refined Hand Kinematics, Vector Spaces & Gesture Pipeline
    # --------------------------------------------------------------------------
    def _smooth_and_analyze_hand(
        self,
        hand_key: str,
        raw_landmarks_norm: List[Tuple[float, float, float]],
        frame_w: int,
        frame_h: int,
        curr_time: float,
        face_anchor_norm: Optional[Tuple[float, float]],
    ) -> Dict[str, Any]:
        """Process hand landmarks with smoothing, kinematics, vector spaces, and gesture engine."""
        smoothed_pts: List[Dict[str, Any]] = []
        rel_face_pts: List[Dict[str, Any]] = []

        prev_list = self.smoothed_landmarks.get(hand_key)

        for lm_id, (rx, ry, rz) in enumerate(raw_landmarks_norm):
            if prev_list is not None and len(prev_list) > lm_id:
                px = prev_list[lm_id]["x"]
                py = prev_list[lm_id]["y"]
                pz = prev_list[lm_id]["z"]
                sx = self.smooth_alpha * rx + (1.0 - self.smooth_alpha) * px
                sy = self.smooth_alpha * ry + (1.0 - self.smooth_alpha) * py
                sz = self.smooth_alpha * rz + (1.0 - self.smooth_alpha) * pz
            else:
                sx, sy, sz = rx, ry, rz

            pt_abs = {
                "id": lm_id,
                "x": float(sx),
                "y": float(sy),
                "z": float(sz),
                "px_x": int(sx * frame_w),
                "px_y": int(sy * frame_h),
            }
            smoothed_pts.append(pt_abs)

            if face_anchor_norm:
                rel_pt = {
                    "id": lm_id,
                    "rel_x": float(sx - face_anchor_norm[0]),
                    "rel_y": float(sy - face_anchor_norm[1]),
                    "rel_z": float(sz),
                }
            else:
                rel_pt = {"id": lm_id, "rel_x": None, "rel_y": None, "rel_z": None}
            rel_face_pts.append(rel_pt)

        self.smoothed_landmarks[hand_key] = smoothed_pts

        # 1. Compute Wrist Velocity and Motion Speed
        wrist = smoothed_pts[0]
        speed = 0.0
        velocity = (0.0, 0.0, 0.0)
        if hand_key in self.prev_wrist_positions:
            old_x, old_y, old_z, old_t = self.prev_wrist_positions[hand_key]
            dt = curr_time - old_t
            if dt > 0.001:
                vx = (wrist["x"] - old_x) / dt
                vy = (wrist["y"] - old_y) / dt
                vz = (wrist["z"] - old_z) / dt
                velocity = (round(vx, 3), round(vy, 3), round(vz, 3))
                speed = round(math.sqrt(vx * vx + vy * vy + vz * vz), 3)

        self.prev_wrist_positions[hand_key] = (wrist["x"], wrist["y"], wrist["z"], curr_time)

        # 2. Compute Palm Scale for Normalized Distances
        middle_mcp = smoothed_pts[9]
        palm_scale = math.sqrt(
            (wrist["x"] - middle_mcp["x"]) ** 2 + (wrist["y"] - middle_mcp["y"]) ** 2
        ) + 1e-6

        # 3. Pinch Detection (Thumb Tip #4 to Index Tip #8)
        thumb_tip = smoothed_pts[4]
        index_tip = smoothed_pts[8]
        pinch_dist = math.sqrt(
            (thumb_tip["x"] - index_tip["x"]) ** 2
            + (thumb_tip["y"] - index_tip["y"]) ** 2
            + (thumb_tip["z"] - index_tip["z"]) ** 2
        )
        norm_pinch = pinch_dist / palm_scale
        is_pinching = norm_pinch < 0.26

        # 4. Gesture Classification
        gesture_name, fingers_ext = self._classify_gesture(smoothed_pts, norm_pinch, palm_scale)

        # 5. Compute Inter-Digital Vector Spaces
        vector_spaces = self._compute_vector_spaces(smoothed_pts, palm_scale)

        return {
            "wrist_abs": wrist,
            "wrist_rel_to_face": rel_face_pts[0],
            "landmarks_abs": smoothed_pts,
            "landmarks_rel_face": rel_face_pts,
            "velocity": velocity,
            "speed": speed,
            "gesture": gesture_name,
            "is_pinching": is_pinching,
            "pinch_distance": round(norm_pinch, 3),
            "fingers_extended": fingers_ext,
            "vector_space": vector_spaces,
        }

    # --------------------------------------------------------------------------
    # Inference & Coordinate Extraction
    # --------------------------------------------------------------------------
    def process_frame(
        self, bgr_frame: np.ndarray, rgb_frame: np.ndarray, frame_w: int, frame_h: int
    ) -> Tuple[Optional[Dict], List[Dict], Any]:
        """Execute face detection, accurate age estimation, and refined hand tracking."""
        curr_time = time.perf_counter()

        if self.use_legacy_solutions:
            rgb_frame.flags.writeable = False
            face_results = self.face_detector.process(rgb_frame)
            hand_results = self.hands_detector.process(rgb_frame)
            rgb_frame.flags.writeable = True

            face_data = self._extract_face_legacy(face_results, frame_w, frame_h)
            raw_hands_list = self._collect_raw_hands_legacy(hand_results)
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            face_results = self.face_detector.detect(mp_image)
            hand_results = self.hands_detector.detect(mp_image)

            face_data = self._extract_face_tasks(face_results, frame_w, frame_h)
            raw_hands_list = self._collect_raw_hands_tasks(hand_results)

        # 1. Refined Age Prediction
        if face_data and self.enable_age:
            if self.frame_count % self.age_interval == 0 or self.cached_age_data is None:
                self.cached_age_data = self.predict_age(bgr_frame, face_data["bbox_px"])
            face_data["age"] = self.cached_age_data
        elif not face_data:
            self.cached_age_data = None
            self.age_probs_smoothed = None

        # 2. Refined Hand Motion Capture & Kinematics
        face_anchor = face_data["center_norm"] if face_data else None
        active_keys = set()
        hands_data = []

        for idx, (label, score, raw_pts) in enumerate(raw_hands_list):
            hand_key = f"{label}_{idx}"
            active_keys.add(hand_key)

            hand_analytics = self._smooth_and_analyze_hand(
                hand_key=hand_key,
                raw_landmarks_norm=raw_pts,
                frame_w=frame_w,
                frame_h=frame_h,
                curr_time=curr_time,
                face_anchor_norm=face_anchor,
            )

            hands_data.append({
                "index": idx,
                "label": label,
                "score": score,
                **hand_analytics,
            })

        # Purge stale smoothed history for hands that left the frame
        for old_key in list(self.smoothed_landmarks.keys()):
            if old_key not in active_keys:
                del self.smoothed_landmarks[old_key]
                if old_key in self.prev_wrist_positions:
                    del self.prev_wrist_positions[old_key]

        return face_data, hands_data, hand_results

    def _collect_raw_hands_legacy(self, hand_results) -> List[Tuple[str, float, List[Tuple[float, float, float]]]]:
        res = []
        if not hand_results or not hand_results.multi_hand_landmarks:
            return res
        for idx, hand_lms in enumerate(hand_results.multi_hand_landmarks):
            label = "Unknown"
            score = 1.0
            if hand_results.multi_handedness and len(hand_results.multi_handedness) > idx:
                c = hand_results.multi_handedness[idx].classification[0]
                label = c.label
                score = float(c.score)
            pts = [(float(lm.x), float(lm.y), float(lm.z)) for lm in hand_lms.landmark]
            res.append((label, score, pts))
        return res

    def _collect_raw_hands_tasks(self, hand_results) -> List[Tuple[str, float, List[Tuple[float, float, float]]]]:
        res = []
        if not hand_results or not hand_results.hand_landmarks:
            return res
        for idx, hand_lms in enumerate(hand_results.hand_landmarks):
            label = "Unknown"
            score = 1.0
            if hand_results.handedness and len(hand_results.handedness) > idx:
                cat = hand_results.handedness[idx][0]
                label = cat.category_name
                score = float(cat.score)
            pts = [(float(lm.x), float(lm.y), float(lm.z)) for lm in hand_lms]
            res.append((label, score, pts))
        return res

    def _extract_face_legacy(self, face_results, frame_w: int, frame_h: int) -> Optional[Dict]:
        if not face_results or not face_results.detections:
            return None
        detection = face_results.detections[0]
        bbox_rel = detection.location_data.relative_bounding_box

        xmin = int(bbox_rel.xmin * frame_w)
        ymin = int(bbox_rel.ymin * frame_h)
        width = int(bbox_rel.width * frame_w)
        height = int(bbox_rel.height * frame_h)
        cx_norm = bbox_rel.xmin + (bbox_rel.width / 2.0)
        cy_norm = bbox_rel.ymin + (bbox_rel.height / 2.0)

        return {
            "score": float(detection.score[0]) if detection.score else 1.0,
            "bbox_norm": {
                "xmin": float(bbox_rel.xmin),
                "ymin": float(bbox_rel.ymin),
                "width": float(bbox_rel.width),
                "height": float(bbox_rel.height),
            },
            "bbox_px": {"xmin": xmin, "ymin": ymin, "width": width, "height": height},
            "center_norm": (float(cx_norm), float(cy_norm)),
            "center_px": (int(cx_norm * frame_w), int(cy_norm * frame_h)),
            "age": None,
        }

    def _extract_face_tasks(self, face_results, frame_w: int, frame_h: int) -> Optional[Dict]:
        if not face_results or not face_results.detections:
            return None
        detection = face_results.detections[0]
        rect = detection.bounding_box

        if hasattr(rect, "origin_x"):
            xmin, ymin, width, height = rect.origin_x, rect.origin_y, rect.width, rect.height
        else:
            xmin, ymin = rect.left, rect.top
            width, height = rect.right - rect.left, rect.bottom - rect.top

        score = float(detection.categories[0].score) if detection.categories else 1.0
        cx_px = xmin + (width / 2.0)
        cy_px = ymin + (height / 2.0)
        cx_norm = cx_px / float(frame_w)
        cy_norm = cy_px / float(frame_h)

        return {
            "score": score,
            "bbox_norm": {
                "xmin": float(xmin / frame_w),
                "ymin": float(ymin / frame_h),
                "width": float(width / frame_w),
                "height": float(height / frame_h),
            },
            "bbox_px": {
                "xmin": int(xmin),
                "ymin": int(ymin),
                "width": int(width),
                "height": int(height),
            },
            "center_norm": (float(cx_norm), float(cy_norm)),
            "center_px": (int(cx_px), int(cy_px)),
            "age": None,
        }

    # --------------------------------------------------------------------------
    # Motion Capture Telemetry Output
    # --------------------------------------------------------------------------
    def emit_mocap_telemetry(
        self,
        frame_idx: int,
        fps: float,
        face_data: Optional[Dict],
        hands_data: List[Dict],
    ):
        """Stream refined motion capture, vector space angles, and gesture telemetry to console."""
        if self.output_format == "json":
            packet = {
                "frame": frame_idx,
                "timestamp_ms": int(time.time() * 1000),
                "fps": round(fps, 1),
                "face": face_data,
                "hands": hands_data,
            }
            sys.stdout.write(json.dumps(packet) + "\n")
            sys.stdout.flush()
        else:
            face_info = "Face: None"
            if face_data:
                fx, fy = face_data["center_norm"]
                age_str = ""
                if face_data.get("age"):
                    ag = face_data["age"]
                    age_str = f" [Age: ~{ag['expected_age']:.0f}y {ag['bracket']} ({int(ag['confidence']*100)}%)]"
                face_info = f"Face: ({fx:.2f}, {fy:.2f}){age_str}"

            hands_info = []
            if hands_data:
                for h in hands_data:
                    lbl = h["label"]
                    w_abs = h["wrist_abs"]
                    w_rel = h["wrist_rel_to_face"]
                    gesture = h["gesture"]
                    spd = h["speed"]
                    angles = h["vector_space"]["angles_deg"]
                    angle_summary = f"θ[T-I:{angles['thumb_index']}°|I-M:{angles['index_middle']}°|Span:{angles['total_span']}°]"
                    if w_rel["rel_x"] is not None:
                        hands_info.append(
                            f"{lbl} [{gesture}|spd:{spd:.2f}|{angle_summary}]: "
                            f"abs=({w_abs['x']:.2f}, {w_abs['y']:.2f}) "
                            f"rel_face=(x:{w_rel['rel_x']:+.2f}, y:{w_rel['rel_y']:+.2f}, z:{w_rel['rel_z']:+.2f})"
                        )
                    else:
                        hands_info.append(
                            f"{lbl} [{gesture}|{angle_summary}]: abs=({w_abs['x']:.2f}, {w_abs['y']:.2f}) rel_face=(N/A)"
                        )
            else:
                hands_info.append("Hands: None")

            summary = " | ".join(hands_info)
            log_line = f"[Frame {frame_idx:05d} | {fps:4.1f} FPS] {face_info} | {summary}\n"
            sys.stdout.write(log_line)
            sys.stdout.flush()

    # --------------------------------------------------------------------------
    # Visual Overlay & Cyber HUD
    # --------------------------------------------------------------------------
    def draw_overlays(
        self,
        frame: np.ndarray,
        face_data: Optional[Dict],
        hands_data: List[Dict],
    ):
        """
        Render face bounding box, smooth neon skeletal joints,
        inter-digital vector space webbing, and gesture status HUD.
        """
        h, w, _ = frame.shape

        # 1. Face Bounding Box & Age Badge
        if face_data:
            bbox = face_data["bbox_px"]
            x, y, bw, bh = bbox["xmin"], bbox["ymin"], bbox["width"], bbox["height"]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w - 1, x + bw), min(h - 1, y + bh)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 128), 2)
            corner = min(18, bw // 4, bh // 4)
            if corner > 0:
                for (cx_pt, cy_pt, dx, dy) in [
                    (x1, y1, corner, 0), (x1, y1, 0, corner),
                    (x2, y1, -corner, 0), (x2, y1, 0, corner),
                    (x1, y2, corner, 0), (x1, y2, 0, -corner),
                    (x2, y2, -corner, 0), (x2, y2, 0, -corner),
                ]:
                    cv2.line(frame, (cx_pt, cy_pt), (cx_pt + dx, cy_pt + dy), (0, 255, 255), 3)

            conf = int(face_data["score"] * 100)
            label_text = f"Face: {conf}%"
            if face_data.get("age"):
                ag = face_data["age"]
                label_text += f" | Age: ~{ag['expected_age']:.0f} yrs {ag['bracket']} ({int(ag['confidence']*100)}%)"

            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            tag_y1 = max(0, y1 - th - 12)
            cv2.rectangle(frame, (x1, tag_y1), (x1 + tw + 10, y1), (15, 15, 15), -1)
            cv2.rectangle(frame, (x1, tag_y1), (x1 + tw + 10, y1), (0, 255, 128), 1)
            cv2.putText(frame, label_text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 1, cv2.LINE_AA)

            cx, cy = face_data["center_px"]
            cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
            cv2.drawMarker(frame, (cx, cy), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)

        # 2. Refined Hand Skeletal Lines, Vector Spaces & Gesture Badges
        tip_ids = [4, 8, 12, 16, 20]

        for hand_idx, hand in enumerate(hands_data):
            lms = hand["landmarks_abs"]
            label = hand["label"]
            gesture = hand["gesture"]
            is_pinching = hand["is_pinching"]
            angles = hand["vector_space"]["angles_deg"]

            # Draw Relative MoCap Line (Face Center -> Hand Wrist)
            if face_data:
                fcx, fcy = face_data["center_px"]
                wx, wy = hand["wrist_abs"]["px_x"], hand["wrist_abs"]["px_y"]
                cv2.line(frame, (fcx, fcy), (wx, wy), (255, 255, 0), 1, cv2.LINE_AA)

            # Draw Inter-Digital Vector Space Webbing (Fingertip to Fingertip)
            for i in range(len(tip_ids) - 1):
                p1 = (lms[tip_ids[i]]["px_x"], lms[tip_ids[i]]["px_y"])
                p2 = (lms[tip_ids[i + 1]]["px_x"], lms[tip_ids[i + 1]]["px_y"])
                cv2.line(frame, p1, p2, (120, 120, 120), 1, cv2.LINE_AA)

            # Draw Finger Skeletal Lines
            for finger_name, connections in self.FINGER_CHAINS.items():
                color = self.FINGER_COLORS[finger_name]
                thickness = 3 if is_pinching and finger_name in ["thumb", "index"] else 2
                for (start_id, end_id) in connections:
                    pt1 = (lms[start_id]["px_x"], lms[start_id]["px_y"])
                    pt2 = (lms[end_id]["px_x"], lms[end_id]["px_y"])
                    cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)

            # Draw Joints
            for lm in lms:
                px, py = lm["px_x"], lm["px_y"]
                lm_id = lm["id"]
                if lm_id in tip_ids:
                    tip_color = (0, 215, 255) if (is_pinching and lm_id in [4, 8]) else (255, 255, 255)
                    cv2.circle(frame, (px, py), 6, (0, 0, 0), -1)
                    cv2.circle(frame, (px, py), 5, tip_color, -1)
                    cv2.circle(frame, (px, py), 2, (0, 0, 255), -1)
                else:
                    cv2.circle(frame, (px, py), 3, (255, 255, 255), -1)

            # Hand Gesture Badge near Wrist
            w_px, w_py = hand["wrist_abs"]["px_x"], hand["wrist_abs"]["px_y"]
            tag_text = f"{label}: [{gesture}] (spd:{hand['speed']:.2f})"
            (gtw, gth), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            gy1 = min(h - 5, w_py + 25)
            cv2.rectangle(frame, (max(5, w_px - 45), gy1 - gth - 6), (max(5, w_px - 45) + gtw + 10, gy1 + 4), (20, 20, 20), -1)
            cv2.rectangle(frame, (max(5, w_px - 45), gy1 - gth - 6), (max(5, w_px - 45) + gtw + 10, gy1 + 4), (0, 255, 255), 1)
            cv2.putText(frame, tag_text, (max(10, w_px - 40), gy1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

            # 4. Vector Space Mini-HUD Box (Bottom Left/Right)
            hud_y_base = h - 90 - (hand_idx * 65)
            cv2.rectangle(frame, (10, hud_y_base), (280, hud_y_base + 58), (15, 15, 15), -1)
            cv2.rectangle(frame, (10, hud_y_base), (280, hud_y_base + 58), (70, 70, 70), 1)

            v_title = f"{label} Hand Vector Space Angles (deg)"
            cv2.putText(frame, v_title, (16, hud_y_base + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)
            row1 = f"T-I: {angles['thumb_index']:4.1f}° | I-M: {angles['index_middle']:4.1f}°"
            row2 = f"M-R: {angles['middle_ring']:4.1f}° | R-P: {angles['ring_pinky']:4.1f}° | Span: {angles['total_span']:4.1f}°"
            cv2.putText(frame, row1, (16, hud_y_base + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, row2, (16, hud_y_base + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 128), 1, cv2.LINE_AA)

        # 5. Top Cyber HUD Header
        cv2.rectangle(frame, (10, 10), (260, 74), (15, 15, 15), -1)
        cv2.rectangle(frame, (10, 10), (260, 74), (80, 80, 80), 1)

        fps_color = (0, 255, 0) if self.fps >= 25 else (0, 165, 255)
        cv2.putText(frame, f"FPS: {self.fps:4.1f} | Vector MoCap", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, fps_color, 2, cv2.LINE_AA)
        status_text = "Age: EMA ON" if self.enable_age else "Age: OFF"
        cv2.putText(frame, f"{status_text} | Multi-Gesture ON", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, "Focus window & press 'q' to exit", (20, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)

    # --------------------------------------------------------------------------
    # Main Execution Loop
    # --------------------------------------------------------------------------
    def run(self):
        """Execute the real-time webcam feed capture, inference, and display loop."""
        cap = self.init_camera()
        window_title = "Refined Face, Age, Vector Space & Gesture MoCap"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

        print("=" * 80)
        print("Starting Real-Time Face, Age, Vector Space & Multi-Gesture MoCap Pipeline")
        print(f"Webcam Index: {self.camera_id} | Resolution: {self.width}x{self.height}")
        print("Inter-Digital Vector Space: 3D Unit Vectors, Angles (θ), Spans, Palm Normal")
        print("Gestures: OPEN_PALM, FIST, PEACE, OK_SIGN, ROCK_ON, SPIDERMAN, CALL_ME, THUMBS_UP/DN, GUN, PINCH")
        print("Telemetry stream active. Focus the OpenCV window and press 'q' to exit.")
        print("=" * 80)

        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    print("Warning: Dropped camera frame. Retrying...", file=sys.stderr)
                    time.sleep(0.01)
                    continue

                self.frame_count += 1
                fps = self.calculate_fps()
                h, w, _ = frame.shape

                # Horizontal flip for natural selfie mirror view
                frame = cv2.flip(frame, 1)

                # Convert BGR frame to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process face, age, and hand tracking modules
                face_data, hands_data, _ = self.process_frame(frame, rgb_frame, w, h)

                # Emit motion capture telemetry to console
                self.emit_mocap_telemetry(self.frame_count, fps, face_data, hands_data)

                # Render face bounding box, age, hand joints, vector spaces, and HUD
                self.draw_overlays(frame, face_data, hands_data)

                # Display video window
                cv2.imshow(window_title, frame)

                # Break on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\nExit key 'q' pressed. Shutting down cleanly...")
                    break

        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Stopping...")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            if hasattr(self.face_detector, "close"):
                self.face_detector.close()
            if hasattr(self.hands_detector, "close"):
                self.hands_detector.close()
            print("Webcam released and all OpenCV windows destroyed safely.")


def main():
    parser = argparse.ArgumentParser(
        description="Refined Real-Time Face, Age, Vector Space & Multi-Gesture MoCap Tracker using OpenCV and MediaPipe."
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="Capture frame width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Capture frame height (default: 480)")
    parser.add_argument("--fps", type=int, default=30, help="Target camera FPS (default: 30)")
    parser.add_argument("--complexity", type=int, default=0, choices=[0, 1], help="Hand model complexity: 0=ultra low latency, 1=higher precision")
    parser.add_argument("--no-age", action="store_true", help="Disable real-time age classification")
    parser.add_argument("--age-interval", type=int, default=4, help="Compute age every N frames (default: 4 for max FPS)")
    parser.add_argument("--smooth-alpha", type=float, default=0.65, help="Hand landmark smoothing EMA coefficient (0.1=heavy smoothing, 1.0=raw, default: 0.65)")
    parser.add_argument("--format", type=str, default="compact", choices=["compact", "json"], help="Console telemetry format: 'compact' or 'json'")

    args = parser.parse_args()

    tracker = MotionCaptureTracker(
        camera_id=args.camera,
        width=args.width,
        height=args.height,
        fps_target=args.fps,
        model_complexity=args.complexity,
        enable_age=not args.no_age,
        age_interval=args.age_interval,
        smooth_alpha=args.smooth_alpha,
        output_format=args.format,
    )
    tracker.run()


if __name__ == "__main__":
    main()
