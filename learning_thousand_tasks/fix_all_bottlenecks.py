import numpy as np
import os
from pathlib import Path

# Path to your demonstrations
DEMO_DIR = "/home/sci-lab/Documents/greenhouse_project/learning_thousand_tasks/assets/demonstrations"

def fix_bottlenecks():
    # Loop through tasks 001 to 031
    for i in range(1, 32):
        task_name = f"demo_task_full_{i:03d}"
        task_path = os.path.join(DEMO_DIR, task_name)
        
        # Check if folder exists
        if not os.path.exists(task_path):
            print(f"Skipping {task_name} (Folder not found)")
            continue

        # Check for robot_states.npy (Source data)
        states_path = os.path.join(task_path, "robot_states.npy")
        if not os.path.exists(states_path):
            print(f"❌ Error: {task_name} missing robot_states.npy! Cannot generate bottleneck.")
            continue

        # Check if bottleneck already exists
        target_file = os.path.join(task_path, "bottleneck_posevec.npy")
        if os.path.exists(target_file):
            print(f"✅ {task_name} already has bottleneck file.")
            continue

        # --- GENERATE THE FILE ---
        try:
            # Load states (T, 16) or (T, 14)
            states = np.load(states_path)
            
            # Extract Z positions (Assuming Index 9 is Z - standard for this repo)
            # If your array is just [q1..7, x,y,z...], Z might be index 9.
            # Let's try to be smart: usually last 7 are pose, so Z is 3rd from start of pose
            # EE pose usually starts at index 7. So Z is index 9.
            z_values = states[:, 9] 
            
            # Find index of lowest point (picking the object)
            min_z_idx = np.argmin(z_values)
            
            # Extract the 7D pose [x, y, z, qx, qy, qz, qw]
            # Assuming pose is indices 7 to 14
            bottleneck_pose = states[min_z_idx, 7:14]
            
            # Save it
            np.save(target_file, bottleneck_pose)
            
            # Also save the duplicate name just in case
            np.save(os.path.join(task_path, "bottleneck_pose.npy"), bottleneck_pose)
            
            print(f"🛠️ Fixed {task_name}: Generated bottleneck at frame {min_z_idx}")
            
        except Exception as e:
            print(f"❌ Failed to process {task_name}: {e}")

if __name__ == "__main__":
    fix_bottlenecks()