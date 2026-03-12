import os
import glob
import h5py
import numpy as np
import cv2
import re

# --- CONFIGURATION ---
DATA_ROOT = os.path.expanduser("~/Documents/greenhouse_project/learning_thousand_tasks/assets/demonstrations")
OUTPUT_DIR = os.path.expanduser("~/Documents/greenhouse_project/act_data")
TASK_NAME = "demo_task_full" 

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def convert():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    task_folders = sorted([f for f in os.listdir(DATA_ROOT) 
                           if os.path.isdir(os.path.join(DATA_ROOT, f)) and TASK_NAME in f])

    print(f"Found {len(task_folders)} episodes.")

    for i, folder_name in enumerate(task_folders):
        folder_path = os.path.join(DATA_ROOT, folder_name)
        
        try:
            # --- 1. Load Images ---
            image_files = glob.glob(os.path.join(folder_path, "head_*.png"))
            image_files.sort(key=natural_sort_key)
            
            if len(image_files) == 0:
                print(f"Skipping {folder_name}: No images found.")
                continue

            images = []
            for img_f in image_files:
                img = cv2.imread(img_f)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
                images.append(img)
            images = np.array(images)

            # --- 2. Load State (Posevecs) and Twists (Velocities) ---
            pose_path = os.path.join(folder_path, "demo_eef_posevecs.npy")
            
            # Check for BOTH possible twist filenames
            twist_options = ["demo_eef_pose_twists.npy", "demo_eef_twists.npy"]
            twist_path = None
            for option in twist_options:
                test_path = os.path.join(folder_path, option)
                if os.path.exists(test_path):
                    twist_path = test_path
                    break
            
            if not os.path.exists(pose_path) or twist_path is None:
                print(f"Skipping {folder_name}: Missing pose or twist files.")
                continue
                
            qpos = np.load(pose_path)
            eef_twists = np.load(twist_path) # This is likely Dim 7: [vx, vy, vz, wx, wy, wz, grip]
           # --- 3. Sync and Construct Action ---
            min_len = min(len(images), len(qpos), len(eef_twists))
            images = images[:min_len]
            qpos = qpos[:min_len]
            eef_twists = eef_twists[:min_len]

            # Force everything to Dim 7 [6 vels + 1 gripper]
            if eef_twists.shape[1] == 7:
                # File already has gripper, use as is
                actions = eef_twists
            elif eef_twists.shape[1] == 6:
                # File is missing gripper, grab it from qpos
                gripper_column = qpos[:, -1:] 
                actions = np.concatenate([eef_twists, gripper_column], axis=1)
            else:
                # If it's something else (like 8), slice it to 7
                print("error")
                actions = eef_twists[:, :7]

            # --- 4. Save to HDF5 ---
            save_path = os.path.join(OUTPUT_DIR, f"episode_{i}.hdf5")
            with h5py.File(save_path, 'w') as root:
                obs = root.create_group('observations')
                img_grp = obs.create_group('images')
                img_grp.create_dataset('head', data=images)
                
                # Input to policy (Current Pose)
                obs.create_dataset('qpos', data=qpos) 
                # Predicted Output (Velocity Twist + Gripper)
                root.create_dataset('action', data=actions)
                
                root.attrs['sim'] = False

            print(f"Processed: {folder_name} -> episode_{i}.hdf5 (Action Dim: {actions.shape[1]})")

        except Exception as e:
            print(f"Error processing {folder_name}: {e}")

if __name__ == "__main__":
    convert()