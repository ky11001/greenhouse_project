import os
import shutil
import numpy as np

DEMO_DIR = "/home/sci-lab/Documents/greenhouse_project/learning_thousand_tasks/assets/demonstrations"

def fix_filenames():
    count = 0
    # Loop through all folders
    for i in range(32, 42):
        task_name = f"demo_task_full_{i:03d}"
        task_path = os.path.join(DEMO_DIR, task_name)
        
        if not os.path.exists(task_path):
            continue

        # --- FIX 1: DUPLICATE BOTTLENECK ---
        existing_bottleneck = os.path.join(task_path, "bottleneck_pose.npy")
        missing_bottleneck = os.path.join(task_path, "bottleneck_posevec.npy")
        
        if os.path.exists(existing_bottleneck) and not os.path.exists(missing_bottleneck):
            shutil.copy(existing_bottleneck, missing_bottleneck)
            print(f"✅ [Fixed] Created bottleneck_posevec.npy for {task_name}")
            count += 1
        elif os.path.exists(missing_bottleneck):
            print(f"🔹 [Skip] {task_name} already has the correct file.")
            
        # --- CHECK 2: WARN ABOUT MISSING SEGMAP ---
        segmap = os.path.join(task_path, "head_camera_ws_segmap.npy")
        if not os.path.exists(segmap):
            print(f"⚠️  [WARNING] {task_name} is MISSING 'head_camera_ws_segmap.npy'!")

    print(f"\nSummary: Fixed {count} folders.")
    print("If you saw warnings about 'ws_segmap.npy', you MUST generate those masks before preprocessing.")

if __name__ == "__main__":
    fix_filenames()