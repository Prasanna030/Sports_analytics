import cv2
import json
import csv
import os
from pathlib import Path
from ultralytics import YOLO
import numpy as np
from collections import defaultdict

# Configuration
VIDEO_PATH = "/home/prasanna/Downloads/Sports_Analytics/Slow_Motion_Sinner_Serve_720P.mp4"
OUTPUT_DIR = "pose_estimation_results_Serve"
KEYPOINTS_JSON = os.path.join(OUTPUT_DIR, "keypoints.json")
KEYPOINTS_CSV = os.path.join(OUTPUT_DIR, "keypoints.csv")
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "skeleton_overlay.mp4")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load YOLO Pose model
print("Loading YOLO Pose model...")
model = YOLO("yolov8l-pose.pt")

# COCO Keypoint names for reference
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Skeleton connections (pairs of keypoint indices)
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
]

def draw_skeleton(frame, keypoints, confidence_threshold=0.5):
    """
    Draw skeleton on frame with keypoints and connections.
    
    Args:
        frame: Input frame
        keypoints: Array of shape (17, 3) containing [x, y, confidence]
        confidence_threshold: Minimum confidence to draw keypoint
    
    Returns:
        Frame with skeleton drawn
    """
    frame_copy = frame.copy()
    h, w = frame.shape[:2]
    
    # Draw keypoints
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > confidence_threshold:
            x, y = int(x), int(y)
            # Draw circle for keypoint
            cv2.circle(frame_copy, (x, y), 5, (0, 255, 0), -1)
            # Draw keypoint name
            cv2.putText(frame_copy, KEYPOINT_NAMES[i], (x + 10, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    # Draw skeleton connections
    for start_idx, end_idx in SKELETON_CONNECTIONS:
        x1, y1, conf1 = keypoints[start_idx]
        x2, y2, conf2 = keypoints[end_idx]
        
        if conf1 > confidence_threshold and conf2 > confidence_threshold:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.line(frame_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    return frame_copy

def process_video():
    """Process video for pose estimation."""
    
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file '{VIDEO_PATH}' not found!")
        return
    
    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo Details:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total Frames: {total_frames}")
    
    # Setup video writer for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
    
    # Storage for keypoints
    all_keypoints = defaultdict(lambda: defaultdict(list))
    frame_count = 0
    
    print(f"\nProcessing video...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"  Frame {frame_count}/{total_frames}")
        
        # Run pose estimation
        results = model(frame, verbose=False)
        
        # Get keypoints
        if results[0].keypoints is not None:
            keypoints_data = results[0].keypoints.data.cpu().numpy()
            
            # Process each person detected
            for person_idx, person_keypoints in enumerate(keypoints_data):
                all_keypoints[frame_count][person_idx] = person_keypoints.tolist()
                
                # Draw skeleton on frame
                frame = draw_skeleton(frame, person_keypoints)
        
        # Write frame to output video
        out.write(frame)
    
    cap.release()
    out.release()
    
    print(f"\nProcessed {frame_count} frames successfully!")
    
    return all_keypoints, total_frames

def save_keypoints_json(keypoints_data):
    """Save keypoints to JSON file."""
    output_data = {
        "metadata": {
            "total_frames": len(keypoints_data),
            "keypoint_names": KEYPOINT_NAMES,
            "model": "yolov8l-pose"
        },
        "frames": {}
    }
    
    for frame_num, frame_data in keypoints_data.items():
        output_data["frames"][str(frame_num)] = frame_data
    
    with open(KEYPOINTS_JSON, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Saved keypoints to {KEYPOINTS_JSON}")

def save_keypoints_csv(keypoints_data):
    """Save keypoints to CSV file."""
    with open(KEYPOINTS_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        header = ["Frame", "Person", "Keypoint_Index", "Keypoint_Name", "X", "Y", "Confidence"]
        writer.writerow(header)
        
        # Write data
        for frame_num in sorted(keypoints_data.keys()):
            frame_data = keypoints_data[frame_num]
            for person_idx, keypoints in frame_data.items():
                for kp_idx, (x, y, conf) in enumerate(keypoints):
                    writer.writerow([
                        frame_num,
                        person_idx,
                        kp_idx,
                        KEYPOINT_NAMES[kp_idx],
                        round(x, 4),
                        round(y, 4),
                        round(conf, 4)
                    ])
    
    print(f"Saved keypoints to {KEYPOINTS_CSV}")

def main():
    """Main execution."""
    print("=" * 60)
    print("YOLO Pose Estimation for Sports Analytics")
    print("=" * 60)
    
    # Process video
    keypoints_data, total_frames = process_video()
    
    # Save results
    print("\nSaving results...")
    save_keypoints_json(keypoints_data)
    save_keypoints_csv(keypoints_data)
    
    print("\n" + "=" * 60)
    print("Pose Estimation Complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  ✓ Skeleton Overlay Video: {OUTPUT_VIDEO}")
    print(f"  ✓ Keypoints (JSON): {KEYPOINTS_JSON}")
    print(f"  ✓ Keypoints (CSV): {KEYPOINTS_CSV}")
    print(f"\nAll outputs saved in '{OUTPUT_DIR}' directory")

if __name__ == "__main__":
    main()
