"""
Sports Analytics - Pose Metrics Calculator
==========================================
Computes sport-specific metrics from pose estimation keypoints.

Supported Sports:
- Cricket Batting
- Tennis Serve  
- Yoga

Author: Sports Analytics Module
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# ============================================================
# CONFIGURATION - Change this to select sport
# ============================================================
SPORT = "yoga"  # Options: "cricket", "tennis", "yoga"

# Path to keypoints JSON file (update based on your output folder)
KEYPOINTS_PATHS = {
    "cricket": "pose_estimation_results/keypoints.json",
    "tennis": "pose_estimation_results_Tennis/keypoints.json",
    "yoga": "pose_estimation_results_Yoga/keypoints.json"
}

OUTPUT_DIR = f"metrics_results_{SPORT}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# KEYPOINT INDICES (COCO format from YOLO Pose)
# ============================================================
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

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

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
    """
    Calculate angle at p2 formed by points p1-p2-p3.
    Returns angle in degrees.
    """
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
    # Handle zero vectors
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1, 1)  # Handle numerical errors
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

# ============================================================
# CRICKET BATTING METRICS
# ============================================================

def cricket_metrics(frames_data):
    """
    Cricket Batting Metrics
    =======================
    
    Metric 1: ELBOW ANGLE (Front Arm)
    ---------------------------------
    What it measures: The angle at the front elbow during the batting stroke.
    
    Why it matters: A straighter front arm (angle close to 180°) during a 
    cover drive indicates proper technique. It helps transfer power efficiently
    from the body to the bat and through to the ball. A bent elbow can result
    in loss of power and poor ball contact.
    
    Ideal range: 150-180° for a proper drive shot
    
    
    Metric 2: HIP-SHOULDER SEPARATION ANGLE
    ---------------------------------------
    What it measures: The rotational difference between hip line and shoulder line.
    
    Why it matters: This separation creates the "torque" that generates power 
    in batting. When the hips rotate before the shoulders, energy is stored 
    and then released explosively. Elite batsmen like Virat Kohli show 
    significant hip-shoulder separation during powerful shots.
    
    Ideal range: 20-45° of separation during the loading phase
    
    
    Metric 3: STANCE STABILITY (Balance Index)
    ------------------------------------------
    What it measures: How stable and balanced the stance is throughout the shot,
    measured by the variance in hip center position.
    
    Why it matters: A stable base allows for better weight transfer and shot
    control. Too much movement indicates loss of balance which can affect
    timing and power. Good batsmen maintain a solid base while still being
    able to move dynamically.
    
    Ideal: Low variance (< 10 pixels) indicates good stability
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
            
            # Metric 1: Front arm elbow angle (assuming right-handed batsman, front arm is left)
            if l_shoulder[2] > 0.3 and l_elbow[2] > 0.3 and l_wrist[2] > 0.3:
                elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
                elbow_angles.append(elbow_angle)
            
            # Metric 2: Hip-Shoulder separation
            if all(p[2] > 0.3 for p in [l_shoulder, r_shoulder, l_hip, r_hip]):
                shoulder_angle = np.arctan2(r_shoulder[1] - l_shoulder[1], 
                                           r_shoulder[0] - l_shoulder[0])
                hip_angle = np.arctan2(r_hip[1] - l_hip[1], 
                                      r_hip[0] - l_hip[0])
                separation = abs(np.degrees(shoulder_angle - hip_angle))
                hip_shoulder_separations.append(separation)
            
            # Metric 3: Hip center for stability
            if l_hip[2] > 0.3 and r_hip[2] > 0.3:
                hip_center = calculate_midpoint(l_hip, r_hip)
                hip_centers.append(hip_center)
    
    # Calculate statistics
    results = {}
    
    # Elbow Angle Stats
    if elbow_angles:
        results['elbow_angle'] = {
            'mean': np.mean(elbow_angles),
            'max': np.max(elbow_angles),
            'min': np.min(elbow_angles),
            'std': np.std(elbow_angles),
            'data': elbow_angles
        }
        print(f"\n📐 METRIC 1: FRONT ARM ELBOW ANGLE")
        print(f"   Average: {results['elbow_angle']['mean']:.1f}°")
        print(f"   Range: {results['elbow_angle']['min']:.1f}° - {results['elbow_angle']['max']:.1f}°")
        print(f"   → {'✓ Good technique!' if results['elbow_angle']['mean'] > 150 else '⚠ Consider straightening the front arm'}")
    
    # Hip-Shoulder Separation Stats
    if hip_shoulder_separations:
        results['hip_shoulder_separation'] = {
            'mean': np.mean(hip_shoulder_separations),
            'max': np.max(hip_shoulder_separations),
            'data': hip_shoulder_separations
        }
        print(f"\n🔄 METRIC 2: HIP-SHOULDER SEPARATION")
        print(f"   Average: {results['hip_shoulder_separation']['mean']:.1f}°")
        print(f"   Maximum: {results['hip_shoulder_separation']['max']:.1f}°")
        print(f"   → {'✓ Good rotation!' if results['hip_shoulder_separation']['max'] > 20 else '⚠ Try to generate more hip rotation'}")
    
    # Stability Stats
    if hip_centers:
        hip_centers = np.array(hip_centers)
        stability_variance = np.var(hip_centers[:, 0]) + np.var(hip_centers[:, 1])
        
        # Calculate frame-to-frame movement for plotting (1D data)
        frame_movements = []
        for i in range(1, len(hip_centers)):
            movement = np.sqrt((hip_centers[i, 0] - hip_centers[i-1, 0])**2 + 
                              (hip_centers[i, 1] - hip_centers[i-1, 1])**2)
            frame_movements.append(movement)
        
        results['stability'] = {
            'variance': stability_variance,
            'data': frame_movements if frame_movements else [0]  # 1D data for plotting
        }
        print(f"\n⚖️ METRIC 3: STANCE STABILITY")
        print(f"   Variance: {stability_variance:.1f} pixels²")
        print(f"   → {'✓ Stable stance!' if stability_variance < 500 else '⚠ Work on maintaining balance'}")
    
    return results

# ============================================================
# TENNIS SERVE METRICS
# ============================================================

def tennis_metrics(frames_data):
    """
    Tennis Serve Metrics
    ====================
    
    Metric 1: KNEE BEND DEPTH (Loading Phase)
    -----------------------------------------
    What it measures: How much the knees bend during the serve preparation,
    measured as the angle at the knee joint.
    
    Why it matters: A deep knee bend (around 90-120°) allows the player to
    store elastic energy in the legs, which is then released explosively
    during the upward drive. This is crucial for generating power in the serve.
    Insufficient knee bend results in serving primarily with the arm,
    reducing both power and increasing injury risk.
    
    Ideal range: 90-130° at maximum bend (lower angle = deeper bend)
    
    
    Metric 2: SHOULDER ROTATION ANGLE
    ---------------------------------
    What it measures: The rotation of the shoulders relative to the baseline,
    tracking how far the shoulders turn during the trophy position.
    
    Why it matters: Shoulder rotation is essential for generating racquet
    head speed. A good shoulder turn (shoulders perpendicular to baseline)
    creates a longer acceleration path for the racquet. Players like 
    Jannik Sinner achieve significant shoulder rotation for powerful serves.
    
    Ideal range: 70-90° of rotation at trophy position
    
    
    Metric 3: ARM EXTENSION AT CONTACT
    ----------------------------------
    What it measures: How straight the hitting arm is at the moment of 
    ball contact, measured as the elbow angle.
    
    Why it matters: Full arm extension (close to 180°) at contact point
    ensures maximum reach and power transfer. It also indicates proper
    timing - hitting the ball at the highest point. A bent arm at contact
    suggests early contact or poor technique.
    
    Ideal range: 160-180° (straighter is better)
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
            
            # Metric 1: Knee bend (use both knees, take minimum angle = deepest bend)
            knee_angle_samples = []
            if l_hip[2] > 0.3 and l_knee[2] > 0.3 and l_ankle[2] > 0.3:
                angle = calculate_angle(l_hip, l_knee, l_ankle)
                knee_angle_samples.append(angle)
            if r_hip[2] > 0.3 and r_knee[2] > 0.3 and r_ankle[2] > 0.3:
                angle = calculate_angle(r_hip, r_knee, r_ankle)
                knee_angle_samples.append(angle)
            if knee_angle_samples:
                knee_angles.append(min(knee_angle_samples))  # Deepest bend
            
            # Metric 2: Shoulder rotation (angle of shoulder line)
            if l_shoulder[2] > 0.3 and r_shoulder[2] > 0.3:
                # Angle from horizontal (0° = facing camera, 90° = sideways)
                shoulder_angle = abs(np.arctan2(r_shoulder[1] - l_shoulder[1], 
                                               r_shoulder[0] - l_shoulder[0]))
                shoulder_rotations.append(np.degrees(shoulder_angle))
            
            # Metric 3: Arm extension (serving arm - typically right for right-hander)
            if r_shoulder[2] > 0.3 and r_elbow[2] > 0.3 and r_wrist[2] > 0.3:
                extension = calculate_angle(r_shoulder, r_elbow, r_wrist)
                arm_extensions.append(extension)
    
    # Calculate statistics
    results = {}
    
    # Knee Bend Stats
    if knee_angles:
        min_knee_angle = min(knee_angles)  # Deepest bend
        results['knee_bend'] = {
            'min_angle': min_knee_angle,
            'mean': np.mean(knee_angles),
            'data': knee_angles
        }
        print(f"\n🦵 METRIC 1: KNEE BEND DEPTH")
        print(f"   Deepest bend: {min_knee_angle:.1f}°")
        print(f"   Average: {results['knee_bend']['mean']:.1f}°")
        print(f"   → {'✓ Good leg drive!' if min_knee_angle < 140 else '⚠ Try bending knees more for power'}")
    
    # Shoulder Rotation Stats
    if shoulder_rotations:
        max_rotation = max(shoulder_rotations)
        results['shoulder_rotation'] = {
            'max': max_rotation,
            'mean': np.mean(shoulder_rotations),
            'data': shoulder_rotations
        }
        print(f"\n💪 METRIC 2: SHOULDER ROTATION")
        print(f"   Maximum rotation: {max_rotation:.1f}°")
        print(f"   Average: {results['shoulder_rotation']['mean']:.1f}°")
        print(f"   → {'✓ Good shoulder turn!' if max_rotation > 45 else '⚠ Increase shoulder rotation'}")
    
    # Arm Extension Stats
    if arm_extensions:
        max_extension = max(arm_extensions)
        results['arm_extension'] = {
            'max': max_extension,
            'mean': np.mean(arm_extensions),
            'data': arm_extensions
        }
        print(f"\n🎾 METRIC 3: ARM EXTENSION AT CONTACT")
        print(f"   Maximum extension: {max_extension:.1f}°")
        print(f"   Average: {results['arm_extension']['mean']:.1f}°")
        print(f"   → {'✓ Full extension!' if max_extension > 160 else '⚠ Extend arm fully at contact'}")
    
    return results

# ============================================================
# YOGA METRICS
# ============================================================

def yoga_metrics(frames_data):
    """
    Yoga Pose Metrics (Optimized for Handstand & Balance Poses)
    ===========================================================
    
    Metric 1: VERTICAL ALIGNMENT SCORE
    ----------------------------------
    What it measures: How well the body parts stack vertically in a line
    from wrists → shoulders → hips → ankles (for handstand).
    
    Why it matters: Proper alignment is the foundation of a stable handstand.
    When body segments are stacked directly over each other, minimal muscular
    effort is needed to maintain the pose. Poor alignment causes the body to
    constantly fight gravity, leading to fatigue and instability.
    
    Ideal: Score close to 100% (all points in a vertical line)
    
    
    Metric 2: BODY SYMMETRY INDEX
    -----------------------------
    What it measures: How balanced the left and right sides of the body are,
    comparing the positions of corresponding joints.
    
    Why it matters: Symmetry indicates even weight distribution and balanced
    muscle engagement. In yoga, asymmetry can indicate compensation patterns,
    muscle imbalances, or incorrect form. A symmetrical pose is more stable
    and reduces injury risk.
    
    Ideal: Symmetry score > 90% (left and right sides mirror each other)
    
    
    Metric 3: POSE STABILITY (Smoothness Index)
    -------------------------------------------
    What it measures: How steady and still the pose is held over time,
    measured by the frame-to-frame variation in keypoint positions.
    
    Why it matters: A stable, steady hold indicates mastery of the pose,
    proper muscle engagement, and good balance. Excessive movement or
    wobbling suggests the practitioner is struggling to maintain the pose
    or has not yet developed the necessary strength and control.
    
    Ideal: Low variance (< 5 pixels average movement between frames)
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
            
            # Metric 1: Vertical Alignment
            # Calculate horizontal deviation from vertical line
            key_points = [
                calculate_midpoint(l_wrist, r_wrist),
                calculate_midpoint(l_shoulder, r_shoulder),
                calculate_midpoint(l_hip, r_hip),
                calculate_midpoint(l_ankle, r_ankle)
            ]
            
            # Check if all points are valid
            if (l_wrist[2] > 0.3 and r_wrist[2] > 0.3 and 
                l_shoulder[2] > 0.3 and r_shoulder[2] > 0.3):
                x_coords = [p[0] for p in key_points]
                x_deviation = np.std(x_coords)  # Standard deviation in X
                # Convert to a 0-100 score (lower deviation = higher score)
                alignment_score = max(0, 100 - x_deviation)
                alignment_scores.append(alignment_score)
            
            # Metric 2: Symmetry Index
            symmetry_pairs = [
                (l_shoulder, r_shoulder),
                (l_elbow, r_elbow),
                (l_wrist, r_wrist),
                (l_hip, r_hip),
                (l_knee, r_knee),
                (l_ankle, r_ankle)
            ]
            
            # Calculate midline
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
            
            # Metric 3: Stability (frame-to-frame movement)
            current_positions = np.array(keypoints)[:, :2]  # Just x, y
            if prev_keypoints is not None:
                prev_positions = np.array(prev_keypoints)[:, :2]
                movement = np.mean(np.sqrt(np.sum((current_positions - prev_positions)**2, axis=1)))
                frame_movements.append(movement)
            prev_keypoints = keypoints
    
    # Calculate statistics
    results = {}
    
    # Alignment Score Stats
    if alignment_scores:
        results['alignment'] = {
            'mean': np.mean(alignment_scores),
            'max': np.max(alignment_scores),
            'min': np.min(alignment_scores),
            'data': alignment_scores
        }
        print(f"\n📏 METRIC 1: VERTICAL ALIGNMENT SCORE")
        print(f"   Average: {results['alignment']['mean']:.1f}%")
        print(f"   Best: {results['alignment']['max']:.1f}%")
        print(f"   → {'✓ Good alignment!' if results['alignment']['mean'] > 70 else '⚠ Work on stacking body segments'}")
    
    # Symmetry Score Stats
    if symmetry_scores:
        results['symmetry'] = {
            'mean': np.mean(symmetry_scores),
            'data': symmetry_scores
        }
        print(f"\n⚖️ METRIC 2: BODY SYMMETRY INDEX")
        print(f"   Average: {results['symmetry']['mean']:.1f}%")
        print(f"   → {'✓ Well balanced!' if results['symmetry']['mean'] > 80 else '⚠ Focus on even weight distribution'}")
    
    # Stability Stats
    if frame_movements:
        avg_movement = np.mean(frame_movements)
        results['stability'] = {
            'avg_movement': avg_movement,
            'smoothness': max(0, 100 - avg_movement * 2),  # Convert to 0-100 score
            'data': frame_movements
        }
        print(f"\n🧘 METRIC 3: POSE STABILITY (Smoothness)")
        print(f"   Avg frame-to-frame movement: {avg_movement:.1f} pixels")
        print(f"   Smoothness score: {results['stability']['smoothness']:.1f}%")
        print(f"   → {'✓ Very stable!' if avg_movement < 10 else '⚠ Try to minimize movement'}")
    
    return results

# ============================================================
# VISUALIZATION
# ============================================================

def plot_metrics(results, sport):
    """Generate plots for the computed metrics."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'{sport.upper()} - Metrics Over Time', fontsize=14, fontweight='bold')
    
    metric_names = list(results.keys())
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    for i, (metric_name, metric_data) in enumerate(results.items()):
        ax = axes[i]
        data = metric_data.get('data', [])
        
        # Convert to numpy array and ensure it's 1D
        try:
            data = np.array(data).flatten()
        except:
            data = np.array([])
        
        if len(data) > 0:
            # Apply smoothing
            if len(data) > 10:
                smoothed = moving_average(data, window=5)
                frames = range(len(smoothed))
                ax.plot(frames, smoothed, color=colors[i], linewidth=2, label='Smoothed')
                ax.fill_between(frames, smoothed, alpha=0.3, color=colors[i])
            else:
                ax.plot(data, color=colors[i], linewidth=2)
            
            # Add mean line
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
    print(f"\n📊 Metrics plot saved to: {plot_path}")

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
    print(f"📁 Results saved to: {output_path}")

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("\n" + "="*60)
    print("🏆 SPORTS ANALYTICS - POSE METRICS CALCULATOR")
    print("="*60)
    print(f"\n🎯 Selected Sport: {SPORT.upper()}")
    
    # Load keypoints
    keypoints_path = KEYPOINTS_PATHS.get(SPORT, KEYPOINTS_PATHS["yoga"])
    
    if not os.path.exists(keypoints_path):
        print(f"\n❌ Error: Keypoints file not found at {keypoints_path}")
        print("   Please run pose_estimation.py first or update the path.")
        return
    
    print(f"📂 Loading keypoints from: {keypoints_path}")
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
        print(f"❌ Unknown sport: {SPORT}")
        return
    
    # Generate visualizations and save results
    if results:
        plot_metrics(results, SPORT)
        save_results(results, SPORT)
    
    print("\n" + "="*60)
    print("✅ Analysis Complete!")
    print("="*60)

if __name__ == "__main__":
    main()
