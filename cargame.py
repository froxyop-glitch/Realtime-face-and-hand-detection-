# ==============================================================================
# CyberDrive: Real-Time Gesture & MoCap Arcade Racing Game
# Powered by OpenCV, Google MediaPipe, and Deep Learning Age/Hand Tracking.
#
# Controls:
#   - Steering: Hold two hands up like a steering wheel and tilt (or move one hand left/right).
#   - Throttle (GAS): Open Palm (🖐).
#   - Brake: Clench Fist (✊).
#   - Drift: Pinch Fingers (🤏).
#   - Nitro Boost: Thumbs Up (👍) or Peace Sign (✌).
#   - Keyboard Fallback: Left/Right (or A/D) to steer, Up (W) for gas, Down (S) for brake, Space for Nitro.
#   - Restart: Press 'R' key.
#   - Exit: Press 'Q' key.
# ==============================================================================

import argparse
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Re-use the optimized MotionCaptureTracker from tracker.py
from tracker import MotionCaptureTracker


class Particle:
    """Dynamic 2D particle for exhaust fire, tire smoke, and crash sparks."""
    def __init__(self, x: float, y: float, vx: float, vy: float, color: Tuple[int, int, int], size: float, life: float):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = color
        self.size = size
        self.life = life
        self.max_life = life

    def update(self, dt: float) -> bool:
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.life -= dt
        return self.life > 0

    def draw(self, canvas: np.ndarray):
        alpha = max(0.0, self.life / self.max_life)
        cur_size = max(1, int(self.size * alpha))
        px, py = int(self.x), int(self.y)
        h, w, _ = canvas.shape
        if 0 <= px < w and 0 <= py < h:
            cv2.circle(canvas, (px, py), cur_size, self.color, -1)


class TrafficCar:
    """AI traffic vehicle moving on the perspective road."""
    def __init__(self, lane: int, car_type: str = "sports"):
        self.lane = lane  # -1 (left), 0 (center), 1 (right)
        self.z = 0.0      # 0.0 (far horizon) -> 1.0 (player level)
        self.speed = random.uniform(0.18, 0.32)
        self.car_type = car_type
        
        # Color palettes by car type
        if car_type == "sports":
            self.color = (0, 30, 240)       # Crimson Red
            self.accent = (180, 220, 255)
        elif car_type == "muscle":
            self.color = (0, 215, 255)      # Amber Yellow
            self.accent = (40, 40, 40)
        else: # truck
            self.color = (220, 180, 50)     # Neon Cyan
            self.accent = (80, 80, 80)

    def update(self, player_speed_factor: float, dt: float):
        # Moves toward player based on relative speed
        rel_speed = (player_speed_factor - self.speed)
        self.z += rel_speed * dt


class Collectible:
    """Collectible bonuses along the track: Coins, Nitro, and Shield."""
    def __init__(self, lane: int, item_type: str = "coin"):
        self.lane = lane  # -1 (left), 0 (center), 1 (right)
        self.z = 0.0
        self.item_type = item_type
        self.collected = False

    def update(self, player_speed_factor: float, dt: float):
        self.z += player_speed_factor * dt


class CyberDriveGame:
    """
    Complete arcade racing game rendered directly in OpenCV with live webcam gesture control.
    """
    WIDTH = 1280
    HEIGHT = 720

    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id

        # 1. Initialize Motion Capture Tracker
        print("[Game] Initializing MediaPipe MoCap & Age Engine...")
        self.tracker = MotionCaptureTracker(
            camera_id=camera_id,
            width=640,
            height=480,
            fps_target=30,
            model_complexity=0,
            enable_age=True,
            age_interval=6,
            smooth_alpha=0.65,
            output_format="compact",
        )
        try:
            self.cap = self.tracker.init_camera()
        except Exception as e:
            print(f"[Game Warning] {e}")
            print("[Game] Starting in Keyboard Control Mode (no webcam detected).")
            self.cap = None

        # 2. Player State
        self.player_x = 0.0            # -1.0 (far left) to +1.0 (far right)
        self.player_speed = 80.0       # km/h
        self.target_speed = 80.0
        self.max_speed = 180.0
        self.nitro_max_speed = 250.0
        self.nitro_amount = 60.0       # 0 to 100%
        self.is_nitro = False
        self.is_drifting = False
        self.score = 0
        self.distance = 0.0
        self.health = 3
        self.shield_timer = 0.0
        self.game_over = False

        # Input states
        self.steer_input = 0.0         # -1.0 (left) to +1.0 (right)
        self.steer_angle_deg = 0.0
        self.current_action = "CRUISE" # GAS, BRAKE, NITRO, DRIFT, CRUISE
        self.driver_age_text = "Scanning..."

        # 3. Environment & Road Simulation
        self.road_curve = 0.0
        self.target_curve = 0.0
        self.curve_timer = 0.0
        self.road_scroll = 0.0

        # Entities
        self.traffic: List[TrafficCar] = []
        self.collectibles: List[Collectible] = []
        self.particles: List[Particle] = []

        # Timers
        self.traffic_spawn_timer = 0.0
        self.collectible_spawn_timer = 0.0
        self.screen_shake = 0.0
        self.notification_text = "READY PLAYER ONE"
        self.notification_timer = 3.0

        # Performance
        self.prev_frame_time = time.perf_counter()
        self.fps = 30.0

    # --------------------------------------------------------------------------
    # Gesture & Steering Processing
    # --------------------------------------------------------------------------
    def process_controls(self, hands_data: List[Dict], face_data: Optional[Dict], key_code: int):
        """
        Derive virtual steering wheel angle and pedals from hand coordinates and gestures.
        """
        # Update driver age if available
        if face_data and face_data.get("age"):
            ag = face_data["age"]
            self.driver_age_text = f"Age: ~{ag['expected_age']:.0f}y {ag['bracket']}"

        raw_steer = 0.0
        angle_deg = 0.0

        # 1. Dual-Hand Virtual Steering Wheel
        if len(hands_data) >= 2:
            h1, h2 = hands_data[0], hands_data[1]
            # Order hands from left to right on screen
            left_h = h1 if h1["wrist_abs"]["x"] < h2["wrist_abs"]["x"] else h2
            right_h = h2 if h1["wrist_abs"]["x"] < h2["wrist_abs"]["x"] else h1

            lx, ly = left_h["wrist_abs"]["x"], left_h["wrist_abs"]["y"]
            rx, ry = right_h["wrist_abs"]["x"], right_h["wrist_abs"]["y"]

            dx = rx - lx
            dy = ry - ly

            if dx > 0.05:
                # dy < 0 means right hand is higher -> turning clockwise (RIGHT)
                angle_rad = math.atan2(dy, dx)
                angle_deg = math.degrees(angle_rad)
                # Normalize angle to steering input [-1.0, 1.0] (max lock at 35 degrees)
                raw_steer = - (angle_deg / 32.0)
                raw_steer = max(-1.0, min(1.0, raw_steer))
                if abs(raw_steer) < 0.08:
                    raw_steer = 0.0

        # 2. Single-Hand Steering (Horizontal Offset from Center)
        elif len(hands_data) == 1:
            h = hands_data[0]
            hx = h["wrist_abs"]["x"]
            # Center is 0.5; offset determines steering
            raw_steer = (hx - 0.5) / 0.22
            raw_steer = max(-1.0, min(1.0, raw_steer))
            if abs(raw_steer) < 0.08:
                raw_steer = 0.0
            angle_deg = - (raw_steer * 28.0)

        # 3. Keyboard Fallback
        if key_code in [ord("a"), ord("A"), 81, 2424832]: # Left
            raw_steer = -0.9
            angle_deg = 25.0
        elif key_code in [ord("d"), ord("D"), 83, 2555904]: # Right
            raw_steer = 0.9
            angle_deg = -25.0

        # Smooth steering input
        self.steer_input = 0.70 * self.steer_input + 0.30 * raw_steer
        self.steer_angle_deg = 0.75 * self.steer_angle_deg + 0.25 * angle_deg

        # 4. Action & Pedal Detection from Gestures
        gestures = [h.get("gesture", "ACTIVE") for h in hands_data]

        is_nitro_active = any(g in ["THUMBS_UP", "SPIDERMAN", "PEACE"] for g in gestures)
        is_braking = any(g == "FIST" for g in gestures)
        is_drifting = any(g == "PINCH" for g in gestures)
        is_gas = any(g == "OPEN_PALM" for g in gestures)

        # Keyboard pedals fallback
        if key_code in [32]: # Space
            is_nitro_active = True
        elif key_code in [ord("w"), ord("W"), 82, 2490368]: # Up
            is_gas = True
        elif key_code in [ord("s"), ord("S"), 84, 2621440]: # Down
            is_braking = True

        # Resolve state priority
        if is_nitro_active and self.nitro_amount > 2.0:
            self.current_action = "NITRO"
            self.is_nitro = True
            self.is_drifting = False
            self.target_speed = self.nitro_max_speed
            self.nitro_amount = max(0.0, self.nitro_amount - 0.4)
        elif is_braking:
            self.current_action = "BRAKE"
            self.is_nitro = False
            self.is_drifting = False
            self.target_speed = 35.0
        elif is_drifting:
            self.current_action = "DRIFT"
            self.is_nitro = False
            self.is_drifting = True
            self.target_speed = 95.0
        elif is_gas:
            self.current_action = "GAS"
            self.is_nitro = False
            self.is_drifting = False
            self.target_speed = self.max_speed
            self.nitro_amount = min(100.0, self.nitro_amount + 0.08) # Slow recharge on gas
        else:
            self.current_action = "CRUISE"
            self.is_nitro = False
            self.is_drifting = False
            self.target_speed = 85.0
            self.nitro_amount = min(100.0, self.nitro_amount + 0.05)

    # --------------------------------------------------------------------------
    # Game Logic & Physics Update
    # --------------------------------------------------------------------------
    def update(self, dt: float):
        if self.game_over:
            return

        # Smooth speed adjustment (inertia)
        accel_rate = 65.0 if self.is_nitro else 40.0
        if self.player_speed < self.target_speed:
            self.player_speed = min(self.target_speed, self.player_speed + accel_rate * dt)
        else:
            self.player_speed = max(self.target_speed, self.player_speed - 70.0 * dt)

        # Player lateral movement
        drift_mult = 1.4 if self.is_drifting else 1.0
        self.player_x += self.steer_input * (self.player_speed / 100.0) * drift_mult * dt * 2.2
        # Auto-centering pull from road curves
        self.player_x += self.road_curve * (self.player_speed / 100.0) * dt * 0.8
        # Clamp to track boundaries
        self.player_x = max(-1.15, min(1.15, self.player_x))

        # Distance & Score increment
        speed_factor = self.player_speed / 100.0
        self.distance += (self.player_speed * dt * 0.277) # meters
        self.score += int(self.player_speed * dt * 0.5)

        # Road scroll & dynamic curving
        self.road_scroll += speed_factor * dt * 8.0
        self.curve_timer += dt
        if self.curve_timer > 4.0:
            self.curve_timer = 0.0
            self.target_curve = random.choice([-0.45, -0.25, 0.0, 0.25, 0.45])
        self.road_curve = 0.95 * self.road_curve + 0.05 * self.target_curve

        # Shield timer decay
        if self.shield_timer > 0:
            self.shield_timer = max(0.0, self.shield_timer - dt)

        # Notification timer decay
        if self.notification_timer > 0:
            self.notification_timer = max(0.0, self.notification_timer - dt)

        # Screen shake decay
        if self.screen_shake > 0:
            self.screen_shake = max(0.0, self.screen_shake - dt * 25.0)

        # Spawn Traffic
        self.traffic_spawn_timer += dt
        if self.traffic_spawn_timer > max(0.9, 2.2 - (self.score / 8000.0)):
            self.traffic_spawn_timer = 0.0
            lane = random.choice([-1, 0, 1])
            car_type = random.choices(["sports", "muscle", "truck"], weights=[5, 3, 2])[0]
            self.traffic.append(TrafficCar(lane=lane, car_type=car_type))

        # Spawn Collectibles
        self.collectible_spawn_timer += dt
        if self.collectible_spawn_timer > 3.0:
            self.collectible_spawn_timer = 0.0
            lane = random.choice([-1, 0, 1])
            item_type = random.choices(["coin", "nitro", "shield"], weights=[7, 2, 1])[0]
            self.collectibles.append(Collectible(lane=lane, item_type=item_type))

        # Update Traffic
        player_car_box = self._get_player_hitbox()
        remaining_traffic = []
        for car in self.traffic:
            car.update(speed_factor * 0.35, dt)

            # Collision check
            if 0.85 <= car.z <= 1.02:
                car_box = self._get_traffic_hitbox(car)
                if self._check_overlap(player_car_box, car_box):
                    if self.shield_timer > 0:
                        # Shield deflection
                        self._spawn_explosion(player_car_box[0], player_car_box[1], (0, 255, 255), 18)
                        self.score += 300
                        continue
                    else:
                        # Crash!
                        self.health -= 1
                        self.screen_shake = 18.0
                        self.player_speed = 40.0
                        self._spawn_explosion(player_car_box[0], player_car_box[1], (0, 0, 255), 35)
                        self.set_notification("CRASH! -1 LIFE")
                        if self.health <= 0:
                            self.game_over = True
                            self.set_notification("GAME OVER - PRESS 'R' TO RESTART")
                        continue

            # Near-miss bonus
            if 0.98 <= car.z <= 1.05 and abs(car.lane * 0.7 - self.player_x) < 0.45:
                self.score += 5

            if car.z < 1.15:
                remaining_traffic.append(car)
            else:
                self.score += 50  # Successful overtake!
        self.traffic = remaining_traffic

        # Update Collectibles
        remaining_items = []
        for item in self.collectibles:
            item.update(speed_factor * 0.35, dt)
            if 0.85 <= item.z <= 1.02:
                item_x = self.WIDTH // 2 + int(item.lane * 240)
                if abs(item.lane * 0.7 - self.player_x) < 0.28:
                    # Picked up!
                    if item.item_type == "coin":
                        self.score += 200
                        self.set_notification("+200 BONUS POINTS!")
                        self._spawn_explosion(item_x, 560, (0, 215, 255), 15)
                    elif item.item_type == "nitro":
                        self.nitro_amount = min(100.0, self.nitro_amount + 35.0)
                        self.set_notification("+35% NITRO REFILLED!")
                        self._spawn_explosion(item_x, 560, (255, 100, 0), 20)
                    elif item.item_type == "shield":
                        self.shield_timer = 8.0
                        self.set_notification("SHIELD ACTIVATED! (8s)")
                        self._spawn_explosion(item_x, 560, (255, 0, 255), 25)
                    continue

            if item.z < 1.15:
                remaining_items.append(item)
        self.collectibles = remaining_items

        # Particle effects
        px, py = self._get_player_screen_pos()
        # Exhaust fire / nitro trails
        if self.is_nitro:
            for offset in [-16, 16]:
                self.particles.append(Particle(
                    x=px + offset + random.uniform(-2, 2),
                    y=py + 42,
                    vx=random.uniform(-10, 10),
                    vy=random.uniform(120, 220),
                    color=(255, 120, 0) if random.random() > 0.5 else (255, 255, 0),
                    size=random.uniform(5, 9),
                    life=random.uniform(0.15, 0.28),
                ))
        elif self.player_speed > 60:
            for offset in [-14, 14]:
                if random.random() > 0.5:
                    self.particles.append(Particle(
                        x=px + offset,
                        y=py + 40,
                        vx=random.uniform(-5, 5),
                        vy=random.uniform(40, 90),
                        color=(150, 150, 150),
                        size=random.uniform(2, 4),
                        life=random.uniform(0.1, 0.2),
                    ))

        # Drift smoke
        if self.is_drifting or abs(self.steer_input) > 0.65:
            for offset in [-24, 24]:
                self.particles.append(Particle(
                    x=px + offset,
                    y=py + 35,
                    vx=random.uniform(-40, 40),
                    vy=random.uniform(20, 60),
                    color=(200, 200, 200),
                    size=random.uniform(6, 12),
                    life=random.uniform(0.2, 0.4),
                ))

        # Update and cull particles
        self.particles = [p for p in self.particles if p.update(dt)]

    def set_notification(self, text: str, duration: float = 2.0):
        self.notification_text = text
        self.notification_timer = duration

    def reset_game(self):
        self.player_x = 0.0
        self.player_speed = 80.0
        self.target_speed = 80.0
        self.nitro_amount = 60.0
        self.is_nitro = False
        self.is_drifting = False
        self.score = 0
        self.distance = 0.0
        self.health = 3
        self.shield_timer = 0.0
        self.game_over = False
        self.traffic.clear()
        self.collectibles.clear()
        self.particles.clear()
        self.set_notification("GAME RESTARTED! GO!")

    def _spawn_explosion(self, x: float, y: float, color: Tuple[int, int, int], count: int):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(40, 240)
            self.particles.append(Particle(
                x=x, y=y,
                vx=math.cos(angle) * speed,
                vy=math.sin(angle) * speed,
                color=color,
                size=random.uniform(4, 9),
                life=random.uniform(0.25, 0.55),
            ))

    def _get_player_screen_pos(self) -> Tuple[int, int]:
        shake_x = random.uniform(-self.screen_shake, self.screen_shake)
        shake_y = random.uniform(-self.screen_shake, self.screen_shake)
        px = int(self.WIDTH // 2 + (self.player_x * 320) + shake_x)
        py = int(580 + shake_y)
        return px, py

    def _get_player_hitbox(self) -> Tuple[int, int, int, int]:
        px, py = self._get_player_screen_pos()
        return (px, py, 60, 90)

    def _get_traffic_hitbox(self, car: TrafficCar) -> Tuple[int, int, int, int]:
        cx, cy, cw, ch = self._project_road_pos(car.lane * 0.7, car.z, width=70, height=100)
        return (cx, cy, cw, ch)

    @staticmethod
    def _check_overlap(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> bool:
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        return abs(x1 - x2) < (w1 + w2) // 2 and abs(y1 - y2) < (h1 + h2) // 2

    # --------------------------------------------------------------------------
    # Perspective Road Projection & Rendering
    # --------------------------------------------------------------------------
    def _project_road_pos(self, road_x: float, z: float, width: int = 60, height: int = 90) -> Tuple[int, int, int, int]:
        """Convert track coordinates (road_x: -1..1, z: 0..1) into 2D perspective screen coordinates."""
        horizon_y = 260
        bottom_y = 700
        py = int(horizon_y + (bottom_y - horizon_y) * (z ** 1.8))

        # Scale expands as objects approach the camera
        scale = 0.15 + (z ** 1.8) * 0.85
        pw = int(width * scale)
        ph = int(height * scale)

        # Dynamic perspective road width
        road_half_w = 70 + (430 - 70) * (z ** 1.8)
        curve_offset = (z ** 2.0) * self.road_curve * 220
        px = int(self.WIDTH // 2 + (road_x * road_half_w) + curve_offset)

        return px, py, pw, ph

    def render_road(self, canvas: np.ndarray):
        """Draw neon synthwave perspective highway with scrolling curbs and lane markings."""
        # 1. Sky & Horizon Gradients
        cv2.rectangle(canvas, (0, 0), (self.WIDTH, 260), (25, 12, 10), -1)  # Deep cyber purple sky

        # Starfield & Horizon grid lines
        for y_sky in range(40, 260, 30):
            cv2.line(canvas, (0, y_sky), (self.WIDTH, y_sky), (45, 20, 25), 1)
        # Distant Synthwave Sun
        cv2.circle(canvas, (self.WIDTH // 2, 240), 60, (0, 180, 255), -1)
        for band_y in range(210, 270, 8):
            cv2.line(canvas, (self.WIDTH // 2 - 70, band_y), (self.WIDTH // 2 + 70, band_y), (25, 12, 10), 2)

        # 2. Road Surface
        horizon_y = 260
        bottom_y = 720
        curve_shift = int(self.road_curve * 220)

        # Road trapezoid polygon
        road_pts = np.array([
            [self.WIDTH // 2 - 75, horizon_y],
            [self.WIDTH // 2 + 75, horizon_y],
            [self.WIDTH // 2 + 450 + curve_shift, bottom_y],
            [self.WIDTH // 2 - 450 + curve_shift, bottom_y]
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [road_pts], (22, 22, 28)) # Dark asphalt

        # 3. Scrolling Curbs (Neon Cyan & Magenta)
        segments = 24
        for i in range(segments):
            z1 = i / segments
            z2 = (i + 1) / segments
            # Animate stripes based on road_scroll
            color_idx = int((i + self.road_scroll) % 2)
            curb_color = (255, 0, 200) if color_idx == 0 else (255, 255, 0)

            # Left curb segment
            x1_l, y1_l, _, _ = self._project_road_pos(-1.05, z1)
            x2_l, y2_l, _, _ = self._project_road_pos(-1.05, z2)
            cv2.line(canvas, (x1_l, y1_l), (x2_l, y2_l), curb_color, max(2, int(8 * z2)))

            # Right curb segment
            x1_r, y1_r, _, _ = self._project_road_pos(1.05, z1)
            x2_r, y2_r, _, _ = self._project_road_pos(1.05, z2)
            cv2.line(canvas, (x1_r, y1_r), (x2_r, y2_r), curb_color, max(2, int(8 * z2)))

            # Lane Dividers (-0.35 and +0.35)
            if color_idx == 0:
                for lane_mark in [-0.35, 0.35]:
                    lx1, ly1, _, _ = self._project_road_pos(lane_mark, z1)
                    lx2, ly2, _, _ = self._project_road_pos(lane_mark, z2)
                    cv2.line(canvas, (lx1, ly1), (lx2, ly2), (200, 200, 200), max(1, int(4 * z2)))

    def render_player_car(self, canvas: np.ndarray):
        """Render high-detail cyber supercar with neon underglow and dynamic steer tilt."""
        px, py = self._get_player_screen_pos()
        tilt = int(self.steer_input * 12)

        # 1. Neon Underglow
        underglow_color = (255, 100, 0) if self.is_nitro else (255, 255, 0)
        cv2.ellipse(canvas, (px, py + 30), (55, 20), 0, 0, 360, underglow_color, -1)

        # Shield Bubble
        if self.shield_timer > 0:
            bubble_pulse = int(math.sin(time.time() * 10) * 4)
            cv2.ellipse(canvas, (px, py), (65 + bubble_pulse, 70 + bubble_pulse), 0, 0, 360, (255, 0, 255), 3)

        # 2. Wheels
        wheel_color = (30, 30, 30)
        for wx, wy in [(-32, -18), (32, -18), (-34, 25), (34, 25)]:
            cv2.rectangle(canvas, (px + wx - 5, py + wy - 10), (px + wx + 5, py + wy + 10), wheel_color, -1)

        # 3. Main Car Chassis (Futuristic Aerodynamic Wedge)
        body_color = (40, 180, 240) if not self.is_nitro else (0, 200, 255)
        chassis_pts = np.array([
            [px + tilt, py - 46],       # Nose
            [px + 26 + tilt, py - 20],  # Right fender
            [px + 30, py + 36],         # Right rear
            [px + 18, py + 42],         # Exhaust right
            [px - 18, py + 42],         # Exhaust left
            [px - 30, py + 36],         # Left rear
            [px - 26 + tilt, py - 20],  # Left fender
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [chassis_pts], body_color)
        cv2.polylines(canvas, [chassis_pts], True, (255, 255, 255), 2)

        # 4. Windshield & Cockpit
        cockpit_pts = np.array([
            [px + tilt, py - 30],
            [px + 14 + tilt, py - 10],
            [px + 12, py + 15],
            [px - 12, py + 15],
            [px - 14 + tilt, py - 10]
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [cockpit_pts], (20, 20, 30))
        cv2.polylines(canvas, [cockpit_pts], True, (0, 255, 255), 1)

        # 5. Glowing Taillights
        brake_bright = (0, 0, 255) if self.current_action == "BRAKE" else (0, 80, 200)
        cv2.rectangle(canvas, (px - 26, py + 36), (px - 10, py + 41), brake_bright, -1)
        cv2.rectangle(canvas, (px + 10, py + 36), (px + 26, py + 41), brake_bright, -1)

    def render_traffic_and_collectibles(self, canvas: np.ndarray):
        """Render 3D projected traffic cars and powerup orbs."""
        # Sort objects by distance (z) so distant objects are drawn first (painter's algorithm)
        render_queue = []
        for car in self.traffic:
            render_queue.append((car.z, "car", car))
        for item in self.collectibles:
            render_queue.append((item.z, "item", item))

        render_queue.sort(key=lambda obj: obj[0])

        for z, obj_type, obj in render_queue:
            if obj_type == "car":
                cx, cy, cw, ch = self._get_traffic_hitbox(obj)
                # Traffic Chassis
                cv2.rectangle(canvas, (cx - cw // 2, cy - ch // 2), (cx + cw // 2, cy + ch // 2), obj.color, -1)
                cv2.rectangle(canvas, (cx - cw // 2, cy - ch // 2), (cx + cw // 2, cy + ch // 2), (255, 255, 255), max(1, int(2 * z)))
                # Roof
                roof_w = int(cw * 0.7)
                roof_h = int(ch * 0.4)
                cv2.rectangle(canvas, (cx - roof_w // 2, cy - roof_h // 2), (cx + roof_w // 2, cy + roof_h // 2), (30, 30, 35), -1)
                # Red Taillights
                tl_size = max(2, int(6 * z))
                cv2.circle(canvas, (cx - cw // 3, cy + ch // 2 - 4), tl_size, (0, 0, 255), -1)
                cv2.circle(canvas, (cx + cw // 3, cy + ch // 2 - 4), tl_size, (0, 0, 255), -1)

            elif obj_type == "item":
                cx, cy, cw, _ = self._project_road_pos(obj.lane * 0.7, obj.z, width=40, height=40)
                radius = max(3, cw // 2)
                pulse = int(math.sin(time.time() * 8 + obj.z * 10) * 2)
                if obj.item_type == "coin":
                    cv2.circle(canvas, (cx, cy), radius + pulse, (0, 215, 255), -1) # Gold
                    cv2.circle(canvas, (cx, cy), radius, (255, 255, 255), 1)
                elif obj.item_type == "nitro":
                    cv2.rectangle(canvas, (cx - radius, cy - radius), (cx + radius, cy + radius), (255, 120, 0), -1)
                elif obj.item_type == "shield":
                    cv2.circle(canvas, (cx, cy), radius + pulse, (255, 0, 255), -1)

    # --------------------------------------------------------------------------
    # Holographic Steering Wheel & HUD Overlays
    # --------------------------------------------------------------------------
    def render_steering_wheel(self, canvas: np.ndarray):
        """Render a rotating cyberpunk holographic steering wheel matching driver hands."""
        center_x = self.WIDTH // 2
        center_y = 630
        wheel_radius = 58
        angle_rad = math.radians(self.steer_angle_deg)

        # Outer rim
        rim_color = (0, 255, 255) if not self.is_drifting else (0, 165, 255)
        cv2.circle(canvas, (center_x, center_y), wheel_radius, rim_color, 3, cv2.LINE_AA)
        cv2.circle(canvas, (center_x, center_y), 12, (255, 255, 255), -1)

        # Rotating wheel spokes
        for spoke_angle in [0, 120, 240]:
            total_rad = angle_rad + math.radians(spoke_angle)
            end_x = int(center_x + math.cos(total_rad) * wheel_radius)
            end_y = int(center_y + math.sin(total_rad) * wheel_radius)
            cv2.line(canvas, (center_x, center_y), (end_x, end_y), (255, 255, 255), 2, cv2.LINE_AA)

        # Hand grip nodes (representing user's hand placements at 9 and 3 o'clock)
        for grip_offset in [math.pi, 0]:
            gx = int(center_x + math.cos(angle_rad + grip_offset) * (wheel_radius + 4))
            gy = int(center_y + math.sin(angle_rad + grip_offset) * (wheel_radius + 4))
            cv2.circle(canvas, (gx, gy), 6, (0, 255, 0), -1)

        # Digital Steer Angle readout
        cv2.putText(
            canvas, f"{int(self.steer_angle_deg):+d}°",
            (center_x - 18, center_y + wheel_radius + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA
        )

    def render_hud(self, canvas: np.ndarray, pip_frame: Optional[np.ndarray]):
        """Render complete telemetry HUD: Speedometer, Nitro gauge, Score, Health, and PiP Cam."""
        # 1. Speedometer & Action State (Bottom Left)
        cv2.rectangle(canvas, (20, self.HEIGHT - 130), (280, self.HEIGHT - 20), (15, 15, 20), -1)
        cv2.rectangle(canvas, (20, self.HEIGHT - 130), (280, self.HEIGHT - 20), (60, 60, 70), 1)

        spd_color = (0, 255, 255) if not self.is_nitro else (0, 140, 255)
        cv2.putText(canvas, f"{int(self.player_speed)}", (35, self.HEIGHT - 65), cv2.FONT_HERSHEY_SIMPLEX, 1.4, spd_color, 3, cv2.LINE_AA)
        cv2.putText(canvas, "KM/H", (160, self.HEIGHT - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        # Action badge
        action_colors = {
            "NITRO": (0, 165, 255),
            "GAS": (0, 255, 128),
            "BRAKE": (0, 0, 255),
            "DRIFT": (255, 0, 230),
            "CRUISE": (200, 200, 200),
        }
        act_col = action_colors.get(self.current_action, (255, 255, 255))
        cv2.putText(canvas, f"PEDAL: [{self.current_action}]", (35, self.HEIGHT - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, act_col, 2, cv2.LINE_AA)

        # 2. Nitro Meter (Right Bottom)
        cv2.rectangle(canvas, (self.WIDTH - 260, self.HEIGHT - 90), (self.WIDTH - 20, self.HEIGHT - 20), (15, 15, 20), -1)
        cv2.rectangle(canvas, (self.WIDTH - 260, self.HEIGHT - 90), (self.WIDTH - 20, self.HEIGHT - 20), (60, 60, 70), 1)
        cv2.putText(canvas, "NITRO BOOST", (self.WIDTH - 245, self.HEIGHT - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)

        # Nitro fill bar
        bar_w = 220
        fill_w = int((self.nitro_amount / 100.0) * bar_w)
        cv2.rectangle(canvas, (self.WIDTH - 245, self.HEIGHT - 50), (self.WIDTH - 245 + bar_w, self.HEIGHT - 35), (35, 35, 45), -1)
        cv2.rectangle(canvas, (self.WIDTH - 245, self.HEIGHT - 50), (self.WIDTH - 245 + fill_w, self.HEIGHT - 35), (255, 140, 0), -1)

        # 3. Top Banner: Score, Distance & Health (Hearts)
        cv2.rectangle(canvas, (0, 0), (self.WIDTH, 55), (15, 15, 20), -1)
        cv2.line(canvas, (0, 55), (self.WIDTH, 55), (0, 255, 255), 2)

        cv2.putText(canvas, f"SCORE: {self.score:06d}", (30, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"DIST: {int(self.distance)}m", (280, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2, cv2.LINE_AA)

        # Driver Health Hearts
        hearts_str = " ".join(["[♥]" for _ in range(max(0, self.health))])
        cv2.putText(canvas, f"LIVES: {hearts_str}", (480, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 2, cv2.LINE_AA)

        # Driver Profile / Age readout
        cv2.putText(canvas, f"DRIVER: {self.driver_age_text}", (720, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 128), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"FPS: {self.fps:.0f}", (self.WIDTH - 90, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        # 4. In-Game Notification Banner
        if self.notification_timer > 0:
            (tw, th), _ = cv2.getTextSize(self.notification_text, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
            nx = (self.WIDTH - tw) // 2
            cv2.rectangle(canvas, (nx - 15, 140), (nx + tw + 15, 140 + th + 16), (15, 15, 15), -1)
            cv2.rectangle(canvas, (nx - 15, 140), (nx + tw + 15, 140 + th + 16), (0, 255, 255), 2)
            cv2.putText(canvas, self.notification_text, (nx, 140 + th + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)

        # 5. Live Picture-in-Picture Driver Webcam Stream (Top Right)
        if pip_frame is not None:
            pip_w, pip_h = 240, 180
            pip_x = self.WIDTH - pip_w - 20
            pip_y = 70
            resized_pip = cv2.resize(pip_frame, (pip_w, pip_h))
            # Border frame
            cv2.rectangle(canvas, (pip_x - 3, pip_y - 3), (pip_x + pip_w + 3, pip_y + pip_h + 3), (0, 255, 128), 2)
            canvas[pip_y:pip_y + pip_h, pip_x:pip_x + pip_w] = resized_pip

            cv2.putText(canvas, "LIVE DRIVER MOCAP", (pip_x + 5, pip_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 128), 1, cv2.LINE_AA)

    # --------------------------------------------------------------------------
    # Main Execution Loop
    # --------------------------------------------------------------------------
    def run(self):
        window_title = "CyberDrive - Real-Time MoCap & Gesture Racing"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, self.WIDTH, self.HEIGHT)

        print("=" * 75)
        print("Starting CyberDrive Arcade Racer")
        print("Steer by rotating both hands | Open Palm = Gas | Fist = Brake | Thumbs Up = Nitro")
        print("Focus the game window. Press 'r' to restart, 'q' to quit.")
        print("=" * 75)

        canvas = np.zeros((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)

        running = True
        try:
            while running:
                t_start = time.perf_counter()
                dt = t_start - self.prev_frame_time
                self.prev_frame_time = t_start
                dt = min(0.08, max(0.001, dt))  # Clamp dt to prevent tunneling on lag spikes
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)

                # 1. Capture Camera Frame or generate keyboard preview
                if self.cap is not None and self.cap.isOpened():
                    success, cam_frame = self.cap.read()
                    if not success:
                        time.sleep(0.01)
                        continue

                    cam_frame = cv2.flip(cam_frame, 1)
                    h_cam, w_cam, _ = cam_frame.shape
                    rgb_frame = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)

                    # 2. Run MoCap Tracking & Age Detection
                    face_data, hands_data, _ = self.tracker.process_frame(cam_frame, rgb_frame, w_cam, h_cam)

                    # Render MoCap overlays onto the webcam frame for PiP preview
                    self.tracker.draw_overlays(cam_frame, face_data, hands_data)
                else:
                    # Synthetic camera frame for keyboard mode
                    cam_frame = np.zeros((240, 320, 3), dtype=np.uint8)
                    cv2.putText(cam_frame, "KEYBOARD MODE", (35, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(cam_frame, "A / D : Steer", (35, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    cv2.putText(cam_frame, "W / S : Gas / Brake", (35, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    cv2.putText(cam_frame, "Space : Nitro", (35, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    face_data = None
                    hands_data = []

                # 3. Read Key Inputs
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("\nExiting CyberDrive cleanly...")
                    running = False
                    break
                elif key == ord("r"):
                    self.reset_game()

                # Restart on Open Palm gesture if Game Over
                if self.game_over and any(h.get("gesture") == "OPEN_PALM" for h in hands_data):
                    self.reset_game()

                # 4. Compute Driving Controls from Hands / Keyboard
                self.process_controls(hands_data, face_data, key)

                # 5. Physics & Entities Update
                self.update(dt)

                # 6. Render Full 720p Game Canvas
                canvas.fill(0)
                self.render_road(canvas)
                for p in self.particles:
                    p.draw(canvas)
                self.render_traffic_and_collectibles(canvas)
                self.render_player_car(canvas)
                self.render_steering_wheel(canvas)
                self.render_hud(canvas, cam_frame)

                # 7. Display Game Window
                cv2.imshow(window_title, canvas)

        except KeyboardInterrupt:
            print("\nGame interrupted by user.")
        finally:
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
            if hasattr(self.tracker.face_detector, "close"):
                self.tracker.face_detector.close()
            if hasattr(self.tracker.hands_detector, "close"):
                self.tracker.hands_detector.close()
            print("Resources released. Thanks for playing CyberDrive!")


def main():
    parser = argparse.ArgumentParser(description="CyberDrive: Real-Time Gesture & MoCap Racing Game")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    args = parser.parse_args()

    game = CyberDriveGame(camera_id=args.camera)
    game.run()


if __name__ == "__main__":
    main()
