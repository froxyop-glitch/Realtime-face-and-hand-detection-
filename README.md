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
Double-click **`run.bat`** in File Explorer, or from PowerShell:
```powershell
.\run.bat
```

### Option 2: Run via Python / Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python tracker.py
```

### Option 3: Stream Full JSON Telemetry
```powershell
python tracker.py --format json
```
Outputs complete JSON packets per frame containing face coordinates, age distribution, 21 smoothed hand joints, kinematics, gesture classifications, and inter-digital vector space angles.
