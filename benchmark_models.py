"""
Pose Estimation Model Benchmarking for Sports Analytics
========================================================

Compares: YOLO Pose, RTMPose, MediaPipe, OpenPose

Benchmark Metrics:
- Inference Speed (FPS)
- Model Size (MB)
- Detection Quality (Confidence, Stability)
- Memory Usage
- Keypoints Coverage
"""



import cv2
import time
import os
import json
import numpy as np
import psutil
from pathlib import Path


import matplotlib
matplotlib.use('Agg')



VIDEO_PATH = "Virat_Kohli_s_cover_drive_footage_from_stands_480P.mp4"
OUTPUT_DIR = "benchmark_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Number of frames to benchmark (set to None for full video)
MAX_FRAMES = 100


def load_yolo_pose():
    """Load YOLO Pose model."""
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n-pose.pt")  # Nano version for fair comparison
        return model, "YOLOv8-Pose"
    except Exception as e:
        print(f"❌ Failed to load YOLO Pose: {e}")
        return None, None

def load_yolo_pose_large():
    """Load YOLO Pose Large model."""
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8l-pose.pt")  # Large version
        return model, "YOLOv8-Pose-L"
    except Exception as e:
        print(f"❌ Failed to load YOLO Pose Large: {e}")
        return None, None

def load_mediapipe():
    """Load MediaPipe Pose model."""
    try:
        # Check NumPy version compatibility first
        np_version = tuple(map(int, np.__version__.split('.')[:2]))
        if np_version[0] >= 2:
            print(f"❌ MediaPipe requires NumPy <2.0 (you have {np.__version__})")
            print("   To fix: pip install 'numpy<2' (may affect other packages)")
            print("   Skipping MediaPipe benchmark...")
            return None, None
        
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        model = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # 0=lite, 1=full, 2=heavy
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        return model, "MediaPipe-Pose"
    except Exception as e:
        print(f"❌ Failed to load MediaPipe: {e}")
        print("   MediaPipe may have NumPy version incompatibility.")
        return None, None

def load_rtmpose():
    """Load RTMPose model (via MMPose)."""
    try:
        # RTMPose requires mmpose installation
        from mmpose.apis import inference_topdown, init_model
        from mmdet.apis import inference_detector, init_detector
        
        # RTMPose config and checkpoint
        pose_config = 'rtmpose-m_8xb256-420e_coco-256x192.py'
        pose_checkpoint = 'rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192.pth'
        
        model = init_model(pose_config, pose_checkpoint, device='cuda:0')
        return model, "RTMPose-M"
    except Exception as e:
        print(f"❌ Failed to load RTMPose: {e}")
        print("   RTMPose requires mmpose. Install with: pip install mmpose mmdet")
        return None, None

def load_openpose():
    """Load OpenPose model (via OpenCV DNN)."""
    try:
        # OpenPose COCO model
        proto_file = "pose/coco/pose_deploy_linevec.prototxt"
        weights_file = "pose/coco/pose_iter_440000.caffemodel"
        
        if os.path.exists(proto_file) and os.path.exists(weights_file):
            net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)
            return net, "OpenPose-COCO"
        else:
            print(f"❌ OpenPose model files not found")
            print("   Download from: https://github.com/CMU-Perceptual-Computing-Lab/openpose")
            return None, None
    except Exception as e:
        print(f"❌ Failed to load OpenPose: {e}")
        return None, None



def run_yolo_inference(model, frame):
    """Run YOLO Pose inference."""
    results = model(frame, verbose=False)
    keypoints = []
    confidences = []
    
    if results[0].keypoints is not None:
        kpts = results[0].keypoints.data.cpu().numpy()
        for person in kpts:
            keypoints.append(person[:, :2])  # x, y
            confidences.append(person[:, 2])  # confidence
    
    return keypoints, confidences

def run_mediapipe_inference(model, frame):
    """Run MediaPipe Pose inference."""
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.process(rgb_frame)
    
    keypoints = []
    confidences = []
    
    if results.pose_landmarks:
        h, w = frame.shape[:2]
        kpts = []
        confs = []
        for landmark in results.pose_landmarks.landmark:
            kpts.append([landmark.x * w, landmark.y * h])
            confs.append(landmark.visibility)
        keypoints.append(np.array(kpts))
        confidences.append(np.array(confs))
    
    return keypoints, confidences

def run_openpose_inference(model, frame):
    """Run OpenPose inference via OpenCV DNN."""
    h, w = frame.shape[:2]
    
    # Prepare input blob
    inp_height = 368
    inp_width = int((inp_height / h) * w)
    
    blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inp_width, inp_height),
                                  (0, 0, 0), swapRB=False, crop=False)
    model.setInput(blob)
    output = model.forward()
    
    # Parse keypoints
    keypoints = []
    confidences = []
    
    points = []
    confs = []
    for i in range(18):  # COCO has 18 keypoints
        prob_map = output[0, i, :, :]
        min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)
        
        x = (w * point[0]) / output.shape[3]
        y = (h * point[1]) / output.shape[2]
        
        if prob > 0.1:
            points.append([x, y])
            confs.append(prob)
        else:
            points.append([0, 0])
            confs.append(0)
    
    if points:
        keypoints.append(np.array(points))
        confidences.append(np.array(confs))
    
    return keypoints, confidences


def get_model_size(model_name):
    """Get approximate model size in MB."""
    sizes = {
        "YOLOv8-Pose": 6.7,      # yolov8n-pose
        "YOLOv8-Pose-L": 85.3,   # yolov8l-pose
        "MediaPipe-Pose": 3.0,   # Lite model
        "RTMPose-M": 13.6,       # Medium model
        "OpenPose-COCO": 200.0,  # Caffe model
    }
    return sizes.get(model_name, 0)

def get_num_keypoints(model_name):
    """Get number of keypoints detected by each model."""
    keypoints = {
        "YOLOv8-Pose": 17,      # COCO format
        "YOLOv8-Pose-L": 17,    # COCO format
        "MediaPipe-Pose": 33,   # Full body + hands
        "RTMPose-M": 17,        # COCO format
        "OpenPose-COCO": 18,    # COCO + neck
    }
    return keypoints.get(model_name, 0)

def calculate_stability(all_keypoints):
    """
    Calculate keypoint stability (lower variance = more stable).
    Measures frame-to-frame jitter.
    """
    if len(all_keypoints) < 2:
        return 0
    
    movements = []
    for i in range(1, len(all_keypoints)):
        if len(all_keypoints[i]) > 0 and len(all_keypoints[i-1]) > 0:
            # Compare first person detected
            curr = np.array(all_keypoints[i][0])
            prev = np.array(all_keypoints[i-1][0])
            
            if curr.shape == prev.shape:
                diff = np.sqrt(np.sum((curr - prev) ** 2, axis=1))
                movements.append(np.mean(diff))
    
    if movements:
        return np.mean(movements)
    return 0

def benchmark_model(model, model_name, inference_func, video_path, max_frames=None):
    """
    Benchmark a single model.
    
    Returns:
        dict: Benchmark results
    """
    print(f"\n{'='*50}")
    print(f"Benchmarking: {model_name}")
    print(f"{'='*50}")
    
    if model is None:
        return None
    
    cap = cv2.VideoCapture(video_path)
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if max_frames:
        total_frames = min(total_frames, max_frames)
    
    # Metrics storage
    inference_times = []
    all_keypoints = []
    all_confidences = []
    detection_counts = []
    memory_usage = []
    
    frame_count = 0
    
    # Warm-up run
    ret, frame = cap.read()
    if ret:
        _ = inference_func(model, frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    print(f"Processing {total_frames} frames...")
    
    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Measure memory before
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Measure inference time
        start_time = time.time()
        keypoints, confidences = inference_func(model, frame)
        end_time = time.time()
        
        # Measure memory after
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        
        inference_times.append(end_time - start_time)
        all_keypoints.append(keypoints)
        all_confidences.append(confidences)
        detection_counts.append(len(keypoints))
        memory_usage.append(mem_after)
        
        frame_count += 1
        
        if frame_count % 20 == 0:
            print(f"  Frame {frame_count}/{total_frames}")
    
    cap.release()
    
    # Calculate metrics
    avg_inference_time = np.mean(inference_times)
    avg_fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
    
    # Calculate average confidence
    flat_confidences = []
    for conf_list in all_confidences:
        for conf in conf_list:
            flat_confidences.extend(conf.flatten())
    avg_confidence = np.mean(flat_confidences) if flat_confidences else 0
    
    # Calculate detection rate
    detection_rate = sum(1 for d in detection_counts if d > 0) / len(detection_counts) * 100
    
    # Calculate stability
    stability = calculate_stability(all_keypoints)
    
    results = {
        "model_name": model_name,
        "model_size_mb": get_model_size(model_name),
        "num_keypoints": get_num_keypoints(model_name),
        "avg_inference_time_ms": avg_inference_time * 1000,
        "avg_fps": avg_fps,
        "max_fps": 1.0 / min(inference_times) if inference_times else 0,
        "min_fps": 1.0 / max(inference_times) if inference_times else 0,
        "avg_confidence": avg_confidence * 100,  # Convert to percentage
        "detection_rate": detection_rate,
        "stability_score": stability,
        "avg_memory_mb": np.mean(memory_usage),
        "peak_memory_mb": np.max(memory_usage),
        "frames_processed": frame_count
    }
    
    # Print results
    print(f"\n📊 Results for {model_name}:")
    print(f"   Speed: {avg_fps:.1f} FPS (avg), {results['max_fps']:.1f} FPS (max)")
    print(f"   Inference Time: {results['avg_inference_time_ms']:.1f} ms/frame")
    print(f"   Model Size: {results['model_size_mb']:.1f} MB")
    print(f"   Keypoints: {results['num_keypoints']}")
    print(f"   Detection Rate: {detection_rate:.1f}%")
    print(f"   Avg Confidence: {results['avg_confidence']:.1f}%")
    print(f"   Stability (lower=better): {stability:.2f} pixels")
    print(f"   Memory: {results['avg_memory_mb']:.1f} MB (avg), {results['peak_memory_mb']:.1f} MB (peak)")
    
    return results



def create_comparison_table(results_list):
    """Create a formatted comparison table."""
    
    print("\n" + "="*80)
    print("📊 BENCHMARK COMPARISON TABLE")
    print("="*80)
    
    # Header
    headers = ["Metric", *[r["model_name"] for r in results_list if r]]
    
    # Metrics to compare
    metrics = [
        ("Speed (FPS)", "avg_fps", "{:.1f}", "higher"),
        ("Inference (ms)", "avg_inference_time_ms", "{:.1f}", "lower"),
        ("Model Size (MB)", "model_size_mb", "{:.1f}", "lower"),
        ("Keypoints", "num_keypoints", "{}", "info"),
        ("Detection Rate (%)", "detection_rate", "{:.1f}", "higher"),
        ("Confidence (%)", "avg_confidence", "{:.1f}", "higher"),
        ("Stability (px)", "stability_score", "{:.2f}", "lower"),
        ("Memory (MB)", "avg_memory_mb", "{:.1f}", "lower"),
    ]
    
    # Print header
    col_width = 18
    header_line = f"{'Metric':<22}"
    for r in results_list:
        if r:
            header_line += f"{r['model_name']:>{col_width}}"
    print(header_line)
    print("-" * len(header_line))
    
    # Print each metric
    for metric_name, metric_key, fmt, best_type in metrics:
        line = f"{metric_name:<22}"
        values = []
        
        for r in results_list:
            if r:
                val = r.get(metric_key, 0)
                values.append(val)
                line += f"{fmt.format(val):>{col_width}}"
        
        # Mark best value
        if values and best_type != "info":
            if best_type == "higher":
                best_idx = np.argmax(values)
            else:
                best_idx = np.argmin(values)
            # Could add highlighting here
        
        print(line)
    
    print("="*80)

def create_recommendation(results_list):
    """Generate recommendations based on benchmark results."""
    
    print("\n" + "="*80)
    print("🎯 RECOMMENDATIONS FOR SPORTS ANALYTICS")
    print("="*80)
    
    valid_results = [r for r in results_list if r]
    
    if not valid_results:
        print("No valid results to analyze.")
        return
    
    # Find best for each category
    fastest = max(valid_results, key=lambda x: x["avg_fps"])
    smallest = min(valid_results, key=lambda x: x["model_size_mb"])
    most_accurate = max(valid_results, key=lambda x: x["avg_confidence"])
    most_stable = min(valid_results, key=lambda x: x["stability_score"]) if any(r["stability_score"] > 0 for r in valid_results) else None


def save_results(results_list):
    """Save benchmark results to JSON."""
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    output = {
        "benchmark_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "video_file": VIDEO_PATH,
        "max_frames": MAX_FRAMES,
        "results": [convert_to_native(r) for r in results_list if r]
    }
    
    output_path = os.path.join(OUTPUT_DIR, "benchmark_results.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n📁 Results saved to: {output_path}")

def create_bar_chart(results_list):
    """Create visualization of benchmark results."""
    try:
        import matplotlib.pyplot as plt
        plt.switch_backend('Agg') 
        
        valid_results = [r for r in results_list if r]
        if not valid_results:
            return
        
        models = [r["model_name"] for r in valid_results]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Pose Estimation Model Benchmark', fontsize=14, fontweight='bold')
        
        # FPS comparison
        ax1 = axes[0, 0]
        fps_values = [r["avg_fps"] for r in valid_results]
        bars1 = ax1.bar(models, fps_values, color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6'][:len(models)])
        ax1.set_ylabel('FPS')
        ax1.set_title('Inference Speed (Higher = Better)')
        ax1.tick_params(axis='x', rotation=15)
        for bar, val in zip(bars1, fps_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{val:.1f}', ha='center', va='bottom', fontsize=10)
        
        # Model Size comparison
        ax2 = axes[0, 1]
        size_values = [r["model_size_mb"] for r in valid_results]
        bars2 = ax2.bar(models, size_values, color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6'][:len(models)])
        ax2.set_ylabel('Size (MB)')
        ax2.set_title('Model Size (Lower = Better)')
        ax2.tick_params(axis='x', rotation=15)
        for bar, val in zip(bars2, size_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{val:.1f}', ha='center', va='bottom', fontsize=10)
        
        # Confidence comparison
        ax3 = axes[1, 0]
        conf_values = [r["avg_confidence"] for r in valid_results]
        bars3 = ax3.bar(models, conf_values, color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6'][:len(models)])
        ax3.set_ylabel('Confidence (%)')
        ax3.set_title('Detection Confidence (Higher = Better)')
        ax3.tick_params(axis='x', rotation=15)
        for bar, val in zip(bars3, conf_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{val:.1f}', ha='center', va='bottom', fontsize=10)
        
        # Detection Rate comparison
        ax4 = axes[1, 1]
        det_values = [r["detection_rate"] for r in valid_results]
        bars4 = ax4.bar(models, det_values, color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6'][:len(models)])
        ax4.set_ylabel('Detection Rate (%)')
        ax4.set_title('Detection Rate (Higher = Better)')
        ax4.tick_params(axis='x', rotation=15)
        for bar, val in zip(bars4, det_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{val:.1f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        chart_path = os.path.join(OUTPUT_DIR, "benchmark_comparison.png")
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 Comparison chart saved to: {chart_path}")
        
    except ImportError:
        print("⚠️ matplotlib not available for visualization")


def main():
    print("\n" + "="*60)
    print("POSE ESTIMATION MODEL BENCHMARK")
    print("="*60)
    
    if not os.path.exists(VIDEO_PATH):
        print(f"\n❌ Video file not found: {VIDEO_PATH}")
        return
    
    print(f"\n📹 Video: {VIDEO_PATH}")
    print(f"Max frames to process: {MAX_FRAMES or 'All'}")
    
    results_list = []
    
    # Benchmark YOLO Pose (Nano - for speed)
    print("\n" + "-"*60)
    print("Loading YOLO Pose (Nano)...")
    model, name = load_yolo_pose()
    if model:
        result = benchmark_model(model, name, run_yolo_inference, VIDEO_PATH, MAX_FRAMES)
        results_list.append(result)
    
    # Benchmark YOLO Pose (Large - for accuracy)
    print("\n" + "-"*60)
    print("Loading YOLO Pose (Large)...")
    model, name = load_yolo_pose_large()
    if model:
        result = benchmark_model(model, name, run_yolo_inference, VIDEO_PATH, MAX_FRAMES)
        results_list.append(result)
    
    # Benchmark MediaPipe
    print("\n" + "-"*60)
    print("Loading MediaPipe Pose...")
    model, name = load_mediapipe()
    if model:
        result = benchmark_model(model, name, run_mediapipe_inference, VIDEO_PATH, MAX_FRAMES)
        results_list.append(result)
    
    # Benchmark RTMPose (if available)
    print("\n" + "-"*60)
    print("Loading RTMPose...")
    model, name = load_rtmpose()
    if model:
        result = benchmark_model(model, name, None, VIDEO_PATH, MAX_FRAMES)  # Would need custom inference
        results_list.append(result)
    
    # Benchmark OpenPose (if available)
    print("\n" + "-"*60)
    print("Loading OpenPose...")
    model, name = load_openpose()
    if model:
        result = benchmark_model(model, name, run_openpose_inference, VIDEO_PATH, MAX_FRAMES)
        results_list.append(result)
    
    # Generate comparison
    if results_list:
        create_comparison_table(results_list)
        create_recommendation(results_list)
        create_bar_chart(results_list)
        save_results(results_list)
    else:
        print("\n❌ No models were successfully benchmarked.")
    
    print("\n" + "="*60)
    print("✅ Benchmark Complete!")
    print("="*60)

if __name__ == "__main__":
    main()
