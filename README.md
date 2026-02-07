# 🏆 Sports Analytics - Pose Estimation

A pose estimation pipeline for sports analytics using YOLO Pose, with sport-specific metrics for **Cricket**, **Tennis**, and **Yoga**.

---

## 📋 Table of Contents
- [Approach](#approach)
- [Models Used](#models-used)
- [Metrics Defined](#metrics-defined)
- [Observations & Limitations](#observations--limitations)
- [Improvement Plan](#improvement-plan)
- [Project Structure](#project-structure)
- [Usage](#usage)

---

## 🎯 Approach

### Pipeline Overview

```
Video Input → YOLO Pose Detection → Keypoint Extraction → Sport Metrics → Visualization
```

### Steps:
1. **Frame Extraction**: Read video frames using OpenCV
2. **Pose Detection**: Run YOLO Pose model to detect 17 COCO keypoints per person
3. **Skeleton Drawing**: Overlay skeleton connections on frames (with option to hide head keypoints for yoga)
4. **Keypoint Export**: Save frame-wise keypoints to JSON and CSV
5. **Metric Computation**: Calculate sport-specific metrics from keypoint data
6. **Benchmarking**: Compare multiple pose estimation models for production decisions

### Keypoints Used (COCO Format)
```
0: nose, 1-4: eyes/ears, 5-6: shoulders, 7-8: elbows, 
9-10: wrists, 11-12: hips, 13-14: knees, 15-16: ankles
```

---

## 🤖 Models Used

### Primary: YOLOv8 Pose

| Model | Size | FPS | Confidence | Use Case |
|-------|------|-----|------------|----------|
| **YOLOv8n-pose** | 6.7 MB | ~40 FPS | 91.5% | Real-time applications |
| **YOLOv8l-pose** | 85.3 MB | ~5 FPS | 71.7% | High stability tracking |

### Why YOLO Pose?

| Factor | Reasoning |
|--------|-----------|
| **Speed** | Real-time inference (40+ FPS) suitable for live sports analysis |
| **Accuracy** | High detection rate (100%) and confidence scores |
| **Ease of Use** | Single-stage detector, simple API via ultralytics |
| **COCO Format** | Standard 17 keypoints, compatible with most sports metrics |
| **Stability** | Low jitter (~2.5px) for reliable metric computation |

### Benchmark Results

```
Model               FPS     Size    Confidence  Stability
─────────────────────────────────────────────────────────
YOLOv8n-pose       38.6    6.7MB      91.5%     2.72px
YOLOv8l-pose        5.2   85.3MB      71.7%     2.47px
```

**Recommendation**: YOLOv8n-pose for real-time; YOLOv8l-pose for offline analysis requiring smooth trajectories.

---

## 📊 Metrics Defined

### Cricket Batting Metrics

| Metric | Description | Keypoints Used | Why It Matters |
|--------|-------------|----------------|----------------|
| **Elbow Angle** | Bat arm bend angle | Shoulder → Elbow → Wrist | Optimal ~150-170° at impact for power shots |
| **Hip-Shoulder Separation** | Torso rotation angle | Hip line vs Shoulder line | Core power generation, higher = more rotation |
| **Stance Stability** | Frame-to-frame hip movement | Both hips | Lower = more stable base, crucial for balance |

### Tennis Serve Metrics

| Metric | Description | Keypoints Used | Why It Matters |
|--------|-------------|----------------|----------------|
| **Knee Bend Angle** | Leg loading depth | Hip → Knee → Ankle | 90-120° optimal for power generation |
| **Shoulder Rotation** | Upper body coil | Shoulder line angle | Kinetic chain efficiency |
| **Arm Extension** | Reach at contact | Shoulder → Elbow → Wrist | Near 180° for maximum height contact |

### Yoga Pose Metrics

| Metric | Description | Keypoints Used | Why It Matters |
|--------|-------------|----------------|----------------|
| **Vertical Alignment** | Body straightness | Shoulder-Hip-Ankle line | Core stability in standing poses |
| **Symmetry Score** | Left-right balance | Mirror keypoint distances | Balance assessment for poses like Tree |
| **Pose Stability** | Movement over time | All keypoints | Lower = steadier hold, better control |

---

## 🔍 Observations & Limitations

### Observations

| Aspect | Finding |
|--------|---------|
| **Detection Rate** | 100% on clear sports footage |
| **Speed** | Nano model achieves real-time (40 FPS) on CPU |
| **Confidence** | Higher on close-up shots, drops with distance |
| **Stability** | Large model has 10% less jitter than nano |

### Limitations

| Problem | Cause | Impact |
|---------|-------|--------|
| **Jitter** | Low video quality, compression artifacts, motion blur | Noisy angle calculations (±5-10° fluctuation) |
| **Background Detection** | Model detects all humans in frame | Wrong athlete tracked, spectators included |
| **Occlusion** | Body parts hidden by equipment/other players | Missing keypoints, incomplete skeleton |
| **Camera Angle** | Far/oblique angles reduce keypoint visibility | Lower confidence, missed detections |
| **Fast Motion** | Blur during quick movements (bowling, serve) | Temporal inconsistency in keypoints |

---

## 🚀 Improvement Plan

### 1. Video Preprocessing

| Technique | Purpose |
|-----------|---------|
| Temporal denoising | Reduce frame-to-frame noise |
| Super-resolution (Real-ESRGAN) | Upscale low-quality videos |
| Motion deblur | Sharpen fast movements |
| CLAHE | Improve contrast in poor lighting |

### 2. Post-Processing for Jitter

| Method | Description |
|--------|-------------|
| **One Euro Filter** | Adaptive smoothing with minimal lag |
| **Kalman Filter** | Predict keypoint positions, smooth trajectory |
| **Savitzky-Golay** | Polynomial smoothing preserving motion peaks |

### 3. Fine-tuning on Sports Data

**Highest impact improvement** - train on sport-specific videos:

```
Pre-trained YOLO-Pose (COCO) → Fine-tune on Sports Dataset → Optimized Model
```

#### Data Collection Strategy

| Aspect | Specification |
|--------|--------------|
| Volume | 5,000-10,000 annotated frames per sport |
| Diversity | Multiple athletes, body types, lighting conditions |
| Quality Range | Include 480p, 720p, 1080p for robustness |
| Occlusion | 30% samples with partial occlusions |

#### Train/Val/Test Split

| Set | Ratio | Key Rule |
|-----|-------|----------|
| Train | 70% | Split by **athlete**, not frame |
| Validation | 15% | Same athlete never in multiple sets |
| Test | 15% | Stratify by action type, difficulty |

### 4. Evaluation Framework

| Type | Metrics |
|------|---------|
| **Automated** | OKS (Object Keypoint Similarity), PCK@0.1, MPJPE |
| **Temporal** | Jitter score, smoothness, tracking accuracy |
| **Human Evaluation** | Visual quality rating (1-5), A/B comparison |
| **Functional** | "Can we measure elbow angle within 5° accuracy?" |

---

## 📁 Project Structure

```
Sports_Analytics/
├── pose_estimation.py      # Main pose detection script
├── metrics.py              # Sport-specific metrics calculator
├── benchmark_models.py     # Model comparison benchmark
├── README.md               # This file
├── yolov8l-pose.pt         # Large YOLO model weights
├── yolov8n-pose.pt         # Nano YOLO model weights (auto-downloaded)
├── pose_estimation_results/      # Output for cricket video
├── pose_estimation_results_Yoga/ # Output for yoga video
├── metrics_results_yoga/         # Metrics output
└── benchmark_results/            # Benchmark comparison output
    ├── benchmark_results.json
    └── benchmark_comparison.png
```

---

## 🛠️ Usage

### Installation

```bash
pip install ultralytics opencv-python numpy matplotlib
```

### Run Pose Estimation

```bash
# Edit VIDEO_PATH in pose_estimation.py, then:
python pose_estimation.py
```

**Outputs:**
- `skeleton_overlay.mp4` - Video with skeleton overlay
- `keypoints.json` - Frame-wise keypoint data
- `keypoints.csv` - Tabular keypoint export

### Run Metrics Analysis

```bash
# Edit paths in metrics.py for your sport, then:
python metrics.py
```

### Run Model Benchmark

```bash
python benchmark_models.py
```

---

## 📚 References

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [COCO Keypoint Format](https://cocodataset.org/#keypoints-2020)
- [MediaPipe Pose](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)
- Chat GPT

---


