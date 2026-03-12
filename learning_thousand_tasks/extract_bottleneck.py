import numpy as np
import os

# Update this to the folder you are currently processing
TASK_DIR = "/home/sci-lab/Documents/greenhouse_project/learning_thousand_tasks/assets/demonstrations/demo_task_full_006"

def extract_bottleneck():
    # 1. Look for the recorded poses (check common names)
    possible_names = ['ee_pose.npy', 'robot_states.npy', 'cartesian_states.npy']
    states_path = None
    for name in possible_names:
        if os.path.exists(os.path.join(TASK_DIR, name)):
            states_path = os.path.join(TASK_DIR, name)
            break
    
    if not states_path:
        print("✗ Could not find any recorded robot state files. Creating a dummy pose...")
        # Create a dummy [x, y, z, qx, qy, qz, qw]
        posevec = np.array([0.5, 0.0, 0.2, 0, 0, 0, 1]) 
    else:
        states = np.load(states_path)
        print(f"✓ Loaded recorded states from {os.path.basename(states_path)}")
        
        # STRATEGY: Find the frame with the lowest Z value (usually the grasp)
        # Assuming index 2 is Z
        z_values = states[:, 2]
        bottleneck_idx = np.argmin(z_values)
        posevec = states[bottleneck_idx]
        print(f"✓ Found bottleneck at frame {bottleneck_idx} (min Z)")

    # 2. Save with the EXACT name the preprocessor wants
    save_path = os.path.join(TASK_DIR, 'bottleneck_posevec.npy')
    np.save(save_path, posevec)
    print(f"✓ Saved: {save_path}")

if __name__ == "__main__":
    extract_bottleneck()