# WeldGuard-AI: In-Situ Real-Time Welding Surveillance & Automated Defect Mitigation Platform

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.15-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-2.0.0-009688.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**WeldGuard-AI** is a deep learning and computer vision system engineered for in-situ factory surveillance, robotic welding quality assurance, and automated defect mitigation. It provides low-latency edge inference to detect defects, eliminate false glare alarms, and execute automated emergency stops (E-STOP) when cumulative defect stressors breach safety bounds.

---

## 🚀 Key Architectural Innovations

### 1. $T_0$ Environment Baseline Lighting Calibration
* **The Glare Challenge**: Ambient factory shop skylights and reflective steel girders produce massive saturated reflection patches (17,500+ px) that cause naive brightness thresholders to mistake sunlight for the welding arc.
* **The Solution**: During camera initialization ($t \in [0.0\text{s}, 0.5\text{s}]$, when the torch is turned off), WeldGuard-AI constructs an ambient illumination map $\mathcal{B}_{\text{env}}(x, y)$ and dilates a spatial glare exclusion mask $\mathcal{M}_{\text{glare}}$. Sunlight and stationary reflections are permanently excluded from arc and spark evaluation.

### 2. Standby Physical Torch Tip Tracking & Adaptive ROI
* **Standby Mode ($250\times250\text{ px}$)**: Tracks the physical robot blowtorch gooseneck, nozzle tip, and wire stickout as the manipulator approaches the workpiece joint.
* **Welding Mode ($400\times400\text{ px}$)**: Upon arc strike, the ROI smoothly zooms out to capture the electric arc column, molten liquid pool, and radial ballistic spark trajectories.
* **Edge Speedup**: Processing only the localized $400\times400$ ROI yields a **13.6x inference speedup** (evaluating only 7.7% of full-frame pixels).

### 3. Weld Pool Geometry Monitoring (WPGM)
Extracts in-situ liquid pool dimensions via real-time geometric ellipse fitting:
* **Burn-Through Risk**: Detects excessive pool width ($W > \mu_W + 2.5\sigma_W$), indicating impending root blowout or excessive heat input.
* **Lack of Penetration / Fusion**: Detects severe pool constriction ($W < \mu_W - 2.0\sigma_W$), flagging insufficient heat penetration.
* **Humping Bead**: Identifies extreme elongation (Aspect Ratio $L/W > 2.8$), preventing premature bead surface freezing.

### 4. Zero-Flapping Industrial Safety Trip Engine (E-STOP)
In real industrial manufacturing, microsecond or 50ms transient sparks must **not** cause alert flapping between `NOMINAL` and `CRITICAL`.
* **Cumulative Defect Stressor Accumulator**:
  $$S_t = \max\left(0.0, \min\left(S_{\text{bound}}, S_{t-1} + \Delta S_t\right)\right)$$
  - Under normal operation, stress decays smoothly ($\Delta S_t = -1.6$ per frame).
  - Transient 50ms ripples only raise stress slightly ($S_t < 15.0$), keeping status firmly `NOMINAL`.
* **Automated Safety Interlock / E-STOP**:
  - Sustained defects (e.g. burn-through or hydrodynamic turbulence lasting > 1.2s) accumulate stress until $S_t \ge 100.0$.
  - Upon breach, the system automatically halts the process, dispatches an `E_STOP` incident to the MES/SCADA network, and renders a prominent **RED BANNER**:
    `[CRITICAL ERROR: PROCESS HALTED] FAILURE STRESSOR BOUND EXCEEDED (100.0/100.0)`
    `OPERATOR INTERVENTION REQUIRED: RETRY / RESET SYSTEM`

---

## 🛠️ Project Structure

```text
WeldGuard-AI/
├── backend/
│   └── main.py                     # FastAPI REST API & SCADA webhook endpoints
├── data/
│   ├── crane_gallery_stages.xlsx   # 28-Stage Crane Gallery fabrication database
│   ├── telemetry_report_*.json     # Benchmark seam telemetry reports
│   └── manual_telemetry_report_*.json
├── docs/
│   └── v2.0_VIDEO_POC_SPECIFICATION.md
├── models/
│   ├── convlstm_best.pth           # Trained Spatiotemporal ConvLSTM Autoencoder
│   └── convlstm_weldguard_kaggle_t4.pth
├── output/
│   ├── weldguard_investor_demo.mp4 # Full 720p @ 30 FPS Investor Showcase Video
│   └── weldguard_investor_demo.gif # Animated visual preview
├── scripts/
│   ├── generate_investor_demo.py   # Investor demo synthesis script
│   └── reconstruct_test_data.py   # Test dataset & Excel reconstruction
├── src/
│   ├── realtime_surveillance.py    # Main Surveillance & Safety Trip Engine
│   ├── models/conv_lstm.py         # 5D ConvLSTM Recurrent Architecture
│   ├── telemetry_video_analyzer.py # Seam optical analyzer & stability index
│   └── manual_video_analyzer.py    # Manual fabrication & grinding analyzer
└── tests/
    ├── test_surveillance.py        # Safety trip, ConvLSTM & surveillance unit tests
    └── test_telemetry.py           # Backend endpoints & 28-stage database tests
```

---

## ⚡ Quickstart

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
pytest tests/
```

### 3. Launch Surveillance Engine
```bash
python src/realtime_surveillance.py \
  --video data/video_20260908_170143.mp4 \
  --process gmaw \
  --roi-size 400 \
  --stressor-bound 100.0 \
  --output-video output/live_surveillance.mp4
```

### 4. Start FastAPI Telemetry Backend
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
Endpoints:
* `GET /api/health` — GPU acceleration & model readiness status
* `GET /api/v2/stages` — Complete 28-stage fabrication database
* `GET /api/v2/telemetry/robotic` — Seam cycle time, arc duty cycle & stability telemetry
* `POST /api/v2/surveillance/webhook` — Factory MES / SCADA incident webhook

---

## 📄 License
This project is licensed under the MIT License.
