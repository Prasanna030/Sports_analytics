"""Sports Analytics - Pose Metrics Calculator

Computes sport-specific metrics from pose estimation keypoints.
Supported Sports: Cricket, Tennis, Yoga
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Configuration
SPORT = "yoga"  # Options: "cricket", "tennis", "yoga"

KEYPOINTS_PATHS = {
    "cricket": "pose_estimation_results/keypoints.json",
    "tennis": "pose_estimation_results_Tennis/keypoints.json",
    "yoga": "pose_estimation_results_Yoga/keypoints.json"
}

OUTPUT_DIR = f"metrics_results_{SPORT}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# COCO format keypoint indices
KEYPOINTS = {
    "nose": 0,
    "left_eye": 1, "right_eye": 2,
    "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16
}

# Utility functions

def load_keypoints(json_path):
    """Load keypoints from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def get_point(keypoints, name):
    """Get (x, y, confidence) for a keypoint by name."""
    idx = KEYPOINTS[name]
    if idx < len(keypoints):
        return keypoints[idx]
    return [0, 0, 0]

def calculate_angle(p1, p2, p3):
    """Calculate angle at p2 formed by points p1-p2-p3 in degrees."""
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1, 1)
    angle = np.arccos(cos_angle)
    return np.degrees(angle)

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_midpoint(p1, p2):
    """Calculate midpoint between two points."""
    return [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]

def moving_average(data, window=5):
    """Smooth data using moving average."""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')

# Cricket batting metrics

def cricket_metrics(frames_data):
    """Calculate cricket batting metrics:
    1. Elbow angle (front arm)
    2. Hip-shoulder separation angle
    3. Stance stability (balance index)
    """
    
    print("\n" + "="*60)
    print("CRICKET BATTING METRICS ANALYSIS")
    print("="*60)
    
    elbow_angles = []
    hip_shoulder_separations = []
    hip_centers = []
    
    for frame_num, persons in frames_data.items():
        for person_id, keypoints in persons.items():
            # Get relevant keypoints
            l_shoulder = get_point(keypoints, "left_shoulder")
            r_shoulder = get_point(keypoints, "right_shoulder")
            l_elbow = get_point(keypoints, "left_elbow")
            r_elbow = get_point(keypoints, "right_elbow")
            l_wrist = get_point(keypoints, "left_wrist")
            r_wrist = get_point(keypoints, "right_wrist")
            l_hip = get_point(keypoints, "left_hip")
            r_hip = get_point(keypoints, "right_hip")
            
            # Front arm elbow angle
            if l_shoulder[2] > 0.3 and l_elbow[2] > 0.3 and l_wrist[2] > 0.3:
                elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
                elbow_angles.append(elbow_angle)
            
            # Hip-shoulder separation
            if all(p[2] > 0.3 for p in [l_shoulder, r_shoulder, l_hip, r_hip]):
                shoulder_angle = np.arctan2(r_shoulder[1] - l_shoulder[1], 
                                           r_shoulder[0] - l_shoulder[0])
                hip_angle = np.arctan2(r_hip[1] - l_hip[1], 
                                      r_hip[0] - l_hip[0])
                separation = abs(np.degrees(shoulder_angle - hip_angle))
                hip_shoulder_separations.append(separation)
            
            # Hip center for stability
            if l_hip[2] > 0.3 and r_hip[2] > 0.3:
                hip_center = calculate_midpoint(l_hip, r_hip)
                hip_centers.append(hip_center)
    
    results = {}
    
    # Elbow angle stats
    if elbow_angles:
        results['elbow_angle'] = {
            'mean': np.mean(elbow_angles),
            'max': np.max(elbow_angles),
            'min': np.min(elbow_angles),
            'std': np.std(elbow_angles),
            'data': elbow_angles
        }
        print(f"\nMETRIC 1: FRONT ARM ELBOW ANGLE")
        print(f"   Average: {results['elbow_angle']['mean']:.1f}")
        print(f"   Range: {results['elbow_angle']['min']:.1f} - {results['elbow_angle']['max']:.1f}")
        print(f"   {'Good technique!' if results['elbow_angle']['mean'] > 150 else 'Consider straightening the front arm'}")
    
    # Hip-shoulder separation stats
    if hip_shoulder_separations:
        results['hip_shoulder_separation'] = {
            'mean': np.mean(hip_shoulder_separations),
            'max': np.max(hip_shoulder_separations),
            'data': hip_shoulder_separations
        }
        print(f"\nMETRIC 2: HIP-SHOULDER SEPARATION")
        print(f"   Average: {results['hip_shoulder_separation']['mean']:.1f}")
        print(f"   Maximum: {results['hip_shoulder_separation']['max']:.1f}")
        print(f"   {'Good rotation!' if results['hip_shoulder_separation']['max'] > 20 else 'Try to generate more hip rotation'}")
    
    # Stability stats
    if hip_centers:
        hip_centers = np.array(hip_centers)
        stability_variance = np.var(hip_centers[:, 0]) + np.var(hip_centers[:, 1])
        
        frame_movements = []
        for i in range(1, len(hip_centers)):
            movement = np.sqrt((hip_centers[i, 0] - hip_centers[i-1, 0])**2 + 
                              (hip_centers[i, 1] - hip_centers[i-1, 1])**2)
            frame_movements.append(movement)
        
        results['stability'] = {
            'variance': stability_variance,
            'data': frame_movements if frame_movements else [0]
        }
        print(f"\nMETRIC 3: STANCE STABILITY")
        print(f"   Variance: {stability_variance:.1f} pixels squared")
        print(f"   {'Stable stance!' if stability_variance < 500 else 'Work on maintaining balance'}")
    
    return results

# Tennis serve metrics

def tennis_metrics(frames_data):
    """Calculate tennis serve metrics:
    1. Knee bend depth (loading phase)
    2. Shoulder rotation angle
    3. Arm extension at contact
    """
    
    print("\n" + "="*60)
    print("TENNIS SERVE METRICS ANALYSIS")
    print("="*60)
    
    knee_angles = []
    shoulder_rotations = []
    arm_extensions = []
    
    for frame_num, persons in frames_data.items():
        for person_id, keypoints in persons.items():
            # Get relevant keypoints
            l_shoulder = get_point(keypoints, "left_shoulder")
            r_shoulder = get_point(keypoints, "right_shoulder")
            l_elbow = get_point(keypoints, "left_elbow")
            r_elbow = get_point(keypoints, "right_elbow")
            l_wrist = get_point(keypoints, "left_wrist")
            r_wrist = get_point(keypoints, "right_wrist")
            l_hip = get_point(keypoints, "left_hip")
            r_hip = get_point(keypoints, "right_hip")
            l_knee = get_point(keypoints, "left_knee")
            r_knee = get_point(keypoints, "right_knee")
            l_ankle = get_point(keypoints, "left_ankle")
            r_ankle = get_point(keypoints, "right_ankle")
            
            # Knee bend (minimum angle = deepest bend)
            knee_angle_samples = []
            if l_hip[2] > 0.3 and l_knee[2] > 0.3 and l_ankle[2] > 0.3:
                angle = calculate_angle(l_hip, l_knee, l_ankle)
                knee_angle_samples.append(angle)
            if r_hip[2] > 0.3 and r_knee[2] > 0.3 and r_ankle[2] > 0.3:
                angle = calculate_angle(r_hip, r_knee, r_ankle)
                knee_angle_samples.append(angle)
            if knee_angle_samples:
                knee_angles.append(min(knee_angle_samples))
            
            # Shoulder rotation
            if l_shoulder[2] > 0.3 and r_shoulder[2] > 0.3:
                shoulder_angle = abs(np.arctan2(r_shoulder[1] - l_shoulder[1], 
                                               r_shoulder[0] - l_shoulder[0]))
                shoulder_rotations.append(np.degrees(shoulder_angle))
            
            # Arm extension
            if r_shoulder[2] > 0.3 and r_elbow[2] > 0.3 and r_wrist[2] > 0.3:
                extension = calculate_angle(r_shoulder, r_elbow, r_wrist)
                arm_extensions.append(extension)
    
    results = {}
    
    # Knee bend stats
    if knee_angles:
        min_knee_angle = min(knee_angles)
        results['knee_bend'] = {
            'min_angle': min_knee_angle,
            'mean': np.mean(knee_angles),
            'data': knee_angles
        }
        print(f"\nMETRIC 1: KNEE BEND DEPTH")
        print(f"   Deepest bend: {min_knee_angle:.1f}")
        print(f"   Average: {results['knee_bend']['mean']:.1f}")
        print(f"   {'Good leg drive!' if min_knee_angle < 140 else 'Try bending knees more for power'}")
    
    # Shoulder rotation stats
    if shoulder_rotations:
        max_rotation = max(shoulder_rotations)
        results['shoulder_rotation'] = {
            'max': max_rotation,
            'mean': np.mean(shoulder_rotations),
            'data': shoulder_rotations
        }
        print(f"\nMETRIC 2: SHOULDER ROTATION")
        print(f"   Maximum rotation: {max_rotation:.1f}")
        print(f"   Average: {results['shoulder_rotation']['mean']:.1f}")
        print(f"   {'Good shoulder turn!' if max_rotation > 45 else 'Increase shoulder rotation'}")
    
    # Arm extension stats
    if arm_extensions:
        max_extension = max(arm_extensions)
        results['arm_extension'] = {
            'max': max_extension,
            'mean': np.mean(arm_extensions),
            'data': arm_extensions
        }
        print(f"\nMETRIC 3: ARM EXTENSION AT CONTACT")
        print(f"   Maximum extension: {max_extension:.1f}")
        print(f"   Average: {results['arm_extension']['mean']:.1f}")
        print(f"   {'Full extension!' if max_extension > 160 else 'Extend arm fully at contact'}")
    
    return results

# Yoga pose metrics

def yoga_metrics(frames_data):
    """Calculate yoga pose metrics:
    1. Vertical alignment score
    2. Body symmetry index
    3. Pose stability (smoothness index)
    """
    
    print("\n" + "="*60)
    print("YOGA POSE METRICS ANALYSIS")
    print("="*60)
    
    alignment_scores = []
    symmetry_scores = []
    frame_positions = []  # For stability calculation
    
    prev_keypoints = None
    frame_movements = []
    
    for frame_num, persons in sorted(frames_data.items(), key=lambda x: int(x[0])):
        for person_id, keypoints in persons.items():
            # Get relevant keypoints
            l_shoulder = get_point(keypoints, "left_shoulder")
            r_shoulder = get_point(keypoints, "right_shoulder")
            l_elbow = get_point(keypoints, "left_elbow")
            r_elbow = get_point(keypoints, "right_elbow")
            l_wrist = get_point(keypoints, "left_wrist")
            r_wrist = get_point(keypoints, "right_wrist")
            l_hip = get_point(keypoints, "left_hip")
            r_hip = get_point(keypoints, "right_hip")
            l_knee = get_point(keypoints, "left_knee")
            r_knee = get_point(keypoints, "right_knee")
            l_ankle = get_point(keypoints, "left_ankle")
            r_ankle = get_point(keypoints, "right_ankle")
            
            # Vertical alignment
            key_points = [
                calculate_midpoint(l_wrist, r_wrist),
                calculate_midpoint(l_shoulder, r_shoulder),
                calculate_midpoint(l_hip, r_hip),
                calculate_midpoint(l_ankle, r_ankle)
            ]
            
            if (l_wrist[2] > 0.3 and r_wrist[2] > 0.3 and 
                l_shoulder[2] > 0.3 and r_shoulder[2] > 0.3):
                x_coords = [p[0] for p in key_points]
                x_deviation = np.std(x_coords)
                alignment_score = max(0, 100 - x_deviation)
                alignment_scores.append(alignment_score)
            
            # Body symmetry
            symmetry_pairs = [
                (l_shoulder, r_shoulder),
                (l_elbow, r_elbow),
                (l_wrist, r_wrist),
                (l_hip, r_hip),
                (l_knee, r_knee),
                (l_ankle, r_ankle)
            ]
            
            midline_x = (l_shoulder[0] + r_shoulder[0]) / 2 if l_shoulder[2] > 0.3 and r_shoulder[2] > 0.3 else 0
            
            symmetry_diffs = []
            for left, right in symmetry_pairs:
                if left[2] > 0.3 and right[2] > 0.3 and midline_x > 0:
                    left_dist = abs(left[0] - midline_x)
                    right_dist = abs(right[0] - midline_x)
                    if max(left_dist, right_dist) > 0:
                        symmetry = min(left_dist, right_dist) / max(left_dist, right_dist)
                        symmetry_diffs.append(symmetry * 100)
            
            if symmetry_diffs:
                symmetry_scores.append(np.mean(symmetry_diffs))
            
            # Stability (frame-to-frame movement)
            current_positions = np.array(keypoints)[:, :2]
            if prev_keypoints is not None:
                prev_positions = np.array(prev_keypoints)[:, :2]
                movement = np.mean(np.sqrt(np.sum((current_positions - prev_positions)**2, axis=1)))
                frame_movements.append(movement)
            prev_keypoints = keypoints
    
    results = {}
    
    # Alignment score stats
    if alignment_scores:
        results['alignment'] = {
            'mean': np.mean(alignment_scores),
            'max': np.max(alignment_scores),
            'min': np.min(alignment_scores),
            'data': alignment_scores
        }
        print(f"\nMETRIC 1: VERTICAL ALIGNMENT SCORE")
        print(f"   Average: {results['alignment']['mean']:.1f}%")
        print(f"   Best: {results['alignment']['max']:.1f}%")
        print(f"   {'Good alignment!' if results['alignment']['mean'] > 70 else 'Work on stacking body segments'}")
    
    # Symmetry score stats
    if symmetry_scores:
        results['symmetry'] = {
            'mean': np.mean(symmetry_scores),
            'data': symmetry_scores
        }
        print(f"\nMETRIC 2: BODY SYMMETRY INDEX")
        print(f"   Average: {results['symmetry']['mean']:.1f}%")
        print(f"   {'Well balanced!' if results['symmetry']['mean'] > 80 else 'Focus on even weight distribution'}")
    
    # Stability stats
    if frame_movements:
        avg_movement = np.mean(frame_movements)
        results['stability'] = {
            'avg_movement': avg_movement,
            'smoothness': max(0, 100 - avg_movement * 2),
            'data': frame_movements
        }
        print(f"\nMETRIC 3: POSE STABILITY (Smoothness)")
        print(f"   Avg frame-to-frame movement: {avg_movement:.1f} pixels")
        print(f"   Smoothness score: {results['stability']['smoothness']:.1f}%")
        print(f"   {'Very stable!' if avg_movement < 10 else 'Try to minimize movement'}")
    
    return results

# Visualization

def plot_metrics(results, sport):
    """Generate plots for the computed metrics."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'{sport.upper()} - Metrics Over Time', fontsize=14, fontweight='bold')
    
    metric_names = list(results.keys())
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    for i, (metric_name, metric_data) in enumerate(results.items()):
        ax = axes[i]
        data = metric_data.get('data', [])
        
        try:
            data = np.array(data).flatten()
        except:
            data = np.array([])
        
        if len(data) > 0:
            if len(data) > 10:
                smoothed = moving_average(data, window=5)
                frames = range(len(smoothed))
                ax.plot(frames, smoothed, color=colors[i], linewidth=2, label='Smoothed')
                ax.fill_between(frames, smoothed, alpha=0.3, color=colors[i])
            else:
                ax.plot(data, color=colors[i], linewidth=2)
            
            mean_val = np.mean(data)
            ax.axhline(y=mean_val, color='gray', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.1f}')
            
        ax.set_title(metric_name.replace('_', ' ').title(), fontweight='bold')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Value')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, f'{sport}_metrics_plot.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nMetrics plot saved to: {plot_path}")

def save_results(results, sport):
    """Save results to a JSON file."""
    output = {
        'sport': sport,
        'summary': {}
    }
    
    for metric_name, metric_data in results.items():
        output['summary'][metric_name] = {
            k: v for k, v in metric_data.items() if k != 'data'
        }
    
    output_path = os.path.join(OUTPUT_DIR, f'{sport}_metrics_summary.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to: {output_path}")

# Main execution

def main():
    print("\n" + "="*60)
    print("SPORTS ANALYTICS - POSE METRICS CALCULATOR")
    print("="*60)
    print(f"\nSelected Sport: {SPORT.upper()}")
    
    # Load keypoints
    keypoints_path = KEYPOINTS_PATHS.get(SPORT, KEYPOINTS_PATHS["yoga"])
    
    if not os.path.exists(keypoints_path):
        print(f"\nError: Keypoints file not found at {keypoints_path}")
        print("   Please run pose_estimation.py first or update the path.")
        return
    
    print(f"Loading keypoints from: {keypoints_path}")
    data = load_keypoints(keypoints_path)
    frames_data = data.get('frames', {})
    print(f"   Loaded {len(frames_data)} frames")
    
    # Compute sport-specific metrics
    if SPORT == "cricket":
        results = cricket_metrics(frames_data)
    elif SPORT == "tennis":
        results = tennis_metrics(frames_data)
    elif SPORT == "yoga":
        results = yoga_metrics(frames_data)
    else:
        print(f"Unknown sport: {SPORT}")
        return
    
    # Generate visualizations and save results
    if results:
        plot_metrics(results, SPORT)
        save_results(results, SPORT)
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)

if __name__ == "__main__":
    main()
