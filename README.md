# Refined Real-Time Face, Accurate Age, Vector Space & Multi-Gesture MoCap Tracker

A high-performance Python application built with **OpenCV** and **Google MediaPipe** designed for real-time tracking of face bounding boxes, accurate continuous age estimation, 21 skeletal hand landmarks, **inter-digital vector spaces**, and a comprehensive **hand gesture library**.

---

## Key Features & New Additions

### 1. Inter-Digital Vector Spaces
- **Finger 3D Direction Vectors**:
  - Computes unit vectors $\vec{V}_i = \frac{P_{tip} - P_{mcp}}{||P_{tip} - P_{mcp}||}$ for each individual finger (Thumb, Index, Middle, Ring, Pinky).
- **Inter-Finger Angles ($\theta$) in Degrees**:
  - $\theta_{\text{Thumb-Index}}$
  - $\theta_{\text{Index-Middle}}$
  - $\theta_{\text{Middle-Ring}}$
  - $\theta_{\text{Ring-Pinky}}$
  - $\theta_{\text{Total Span}}$ (Spread between Thumb and Pinky)
- **Inter-Fingertip Normalized Spans**:
  - Euclidean distances between adjacent fingertips normalized by palm scale.
- **3D Palm Plane Normal Vector**:
  - Calculates $\vec{N}_{\text{palm}} = (\vec{P}_{\text{Index\_MCP}} - \vec{P}_{\text{Wrist}}) \times (\vec{P}_{\text{Pinky\_MCP}} - \vec{P}_{\text{Wrist}})$, representing palm orientation in 3D camera space.
- **Vector Space HUD & Webbing**:
  - Visual webbing connecting adjacent fingertips with real-time angular readouts in a bottom HUD module.

### 2. Multi-Gesture Recognition Library
Classifies gestures in real time with high accuracy:
1. **`PEACE` / `VICTORY`**: Index + Middle extended, Ring + Pinky curled, separated by spread angle.
2. **`OK_SIGN`**: Thumb tip and Index tip touching, other 3 fingers extended.
3. **`ROCK_ON`**: Index and Pinky extended, Middle and Ring curled.
4. **`SPIDERMAN` (ILY)**: Thumb, Index, and Pinky extended, Middle and Ring curled.
5. **`CALL_ME` (Shaka)**: Thumb and Pinky extended, other 3 curled.
6. **`THUMBS_UP`**: Thumb extended upward ($y < -0.06$), other 4 fingers curled into a fist.
7. **`THUMBS_DOWN`**: Thumb pointing downward ($y > +0.06$), other 4 fingers curled.
8. **`POINTING`**: Index finger extended, others curled.
9. **`GUN`**: Thumb extended upward, Index extended forward, others curled.
10. **`PINCH`**: Thumb tip and Index tip touching (distance $< 0.26$ palm scale).
11. **`THREE`**: 3 extended fingers.
12. **`FOUR`**: 4 extended fingers with thumb tucked.
13. **`OPEN_PALM`**: All fingers extended wide.
14. **`FIST`**: All fingers curled tightly.

### 3. Accurate Continuous Age Estimation
- **Square Aspect-Ratio Preserving Crop**: Centers square crop with a 35% margin to capture hairline and jawline without distortion.
- **Softmax Probability EMA Smoothing**: Eliminates single-frame lighting/pose fluctuations.
- **Continuous Expected Age**: Computes mathematical expected age ($E[\text{Age}] = \sum P_i \times \text{Midpoint}_i$) alongside the age bracket.

### 4. Anti-Jitter Smoothing & Kinematics
- **EMA Landmark Smoothing ($\alpha = 0.65$)**: Removes high-frequency camera noise and tremor.
- **Velocity Vectors & Speed**: Computes real-time hand speed and motion dynamics.

---

## Setup & Running

### Option 1: Quick Launch (Batch File)
From anywhere in PowerShell or CMD:
```powershell
& "C:\Users\shoho\.gemini\antigravity\scratch\motion_tracker\run.bat"
```
*(Or double-click **`run.bat`** in File Explorer)*

### Option 2: Run via PowerShell
```powershell
cd C:\Users\shoho\.gemini\antigravity\scratch\motion_tracker
.\.venv\Scripts\python.exe tracker.py
```

### Option 3: Stream Full JSON Telemetry
```powershell
python tracker.py --format json
```
Outputs complete JSON packets per frame containing face coordinates, age distribution, 21 smoothed hand joints, kinematics, gesture classifications, and inter-digital vector space angles.

---

## 🏎️ CyberDrive: Gesture & MoCap Arcade Racing Game

A complete 720p 3D-perspective arcade racing game built on top of the MoCap and Age detection pipeline!

### Controls & Gesture Mechanics
- **Virtual Steering Wheel**: Hold both hands up in the air as if gripping a steering wheel and tilt left/right. The game computes $\theta = \arctan2(\Delta y, \Delta x)$ between your wrists to turn your car with precision. (Single-hand horizontal offset also supported!)
- **Accelerate (Gas)**: Open Palm (🖐)
- **Brake / Reverse**: Clench Fist (✊)
- **Nitro Boost**: Thumbs Up (👍) or Peace Sign (✌)
- **Drift / Power Slide**: Pinch Fingers (🤏)
- **Keyboard Fallback**: `A` / `D` or Arrow keys to steer, `W` for gas, `S` for brake, `Space` for Nitro.
- **Restart**: Press `R` or flash an Open Palm when crashed.
- **Exit**: Press `Q`.

### Features
- **Dynamic 3D-Perspective Highway**: Curved roads, hill crests, horizon glow, and asphalt markings.
- **AI Traffic Vehicles**: Sports cars, muscle cars, and trucks navigating lanes.
- **Bonus Collectibles**: Gold Coins (+100 pts), Nitro Canisters (+35% boost), and Forcefield Shields (invulnerability).
- **Interactive Holographic Steering Wheel**: 3D-rendered steering wheel rotates dynamically with your hand tilt angle.
- **Live MoCap PiP Driver Display**: Picture-in-picture stream displaying your face, real-time age badge, hand skeletal bones, and active gesture telemetry.
- **Particle System**: Exhaust flames, tire smoke on drifting, and sparks on collision.

### How to Run the Car Game
```powershell
# Quick Launch via Batch
& "C:\Users\shoho\.gemini\antigravity\scratch\motion_tracker\run_cargame.bat"

# Or via Python in terminal
cd C:\Users\shoho\.gemini\antigravity\scratch\motion_tracker
.\.venv\Scripts\python.exe cargame.py
```

