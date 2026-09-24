import sys
import numpy as np

# Import CyberDriveGame
from cargame import CyberDriveGame, TrafficCar, Collectible

def test_simulation():
    print("[Test] Initializing CyberDriveGame headless...")
    # Instantiate CyberDriveGame without opening camera
    game = CyberDriveGame.__new__(CyberDriveGame)
    game.WIDTH = 1280
    game.HEIGHT = 720
    game.player_x = 0.0
    game.player_speed = 80.0
    game.target_speed = 80.0
    game.max_speed = 180.0
    game.nitro_max_speed = 250.0
    game.nitro_amount = 60.0
    game.is_nitro = False
    game.is_drifting = False
    game.score = 0
    game.distance = 0.0
    game.health = 3
    game.shield_timer = 0.0
    game.game_over = False
    game.steer_input = 0.0
    game.steer_angle_deg = 0.0
    game.current_action = "CRUISE"
    game.driver_age_text = "Test Driver"
    game.road_curve = 0.0
    game.target_curve = 0.0
    game.curve_timer = 0.0
    game.road_scroll = 0.0
    game.traffic = []
    game.collectibles = []
    game.particles = []
    game.traffic_spawn_timer = 0.0
    game.collectible_spawn_timer = 0.0
    game.screen_shake = 0.0
    game.notification_text = "TESTING"
    game.notification_timer = 2.0
    game.prev_frame_time = 0.0
    game.fps = 30.0

    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Test projection with negative, zero, and boundary z values
    test_z_values = [-1.0, -0.5, -0.001, 0.0, 0.0001, 0.5, 1.0, 1.15, 2.0]
    for z in test_z_values:
        px, py, pw, ph = game._project_road_pos(0.0, z)
        assert isinstance(px, int) and isinstance(py, int), f"Projection failed for z={z}"

    print("[Test] Direct projection test passed for all boundary and negative z values.")

    # Simulate 500 frames of gameplay with varying inputs (braking, speeding, drifting)
    import random
    actions = ["GAS", "BRAKE", "NITRO", "DRIFT", "CRUISE"]
    for frame in range(500):
        dt = 0.033
        action = random.choice(actions)
        if action == "GAS":
            game.target_speed = 170.0
        elif action == "BRAKE":
            game.target_speed = 20.0  # Heavy braking -> causes relative negative speed
        elif action == "NITRO":
            game.target_speed = 240.0
        elif action == "DRIFT":
            game.target_speed = 90.0
            game.is_drifting = True
        else:
            game.target_speed = 80.0
            game.is_drifting = False

        game.steer_input = random.uniform(-1.0, 1.0)
        game.steer_angle_deg = -game.steer_input * 30.0

        # Run physics update
        game.update(dt)

        # Run full render pipeline to test drawing math
        canvas.fill(0)
        game.render_road(canvas)
        for p in game.particles:
            p.draw(canvas)
        game.render_traffic_and_collectibles(canvas)
        game.render_player_car(canvas)
        game.render_steering_wheel(canvas)

    print(f"[Test] Successfully simulated 500 frames! Final score: {game.score}, Distance: {game.distance:.1f}m")

if __name__ == "__main__":
    test_simulation()
