import numpy as np
import os

# CONFIGURATION
DEMO_ROOT = "/home/sci-lab/Documents/greenhouse_project/learning_thousand_tasks/assets/demonstrations"
START_TASK = 1
END_TASK = 31

def apply_precise_intrinsics():
    # Your specific RealSense calibration
    # fx, fy = ~386 (Wider FOV), cx, cy = ~Center
    precise_intrinsics = np.array([
        [386.02066040,   0.0,          321.55554199],
        [  0.0,          385.41479492, 240.14253235],
        [  0.0,          0.0,          1.0         ]
    ], dtype=np.float64)

    print(f"Applying precise intrinsics to tasks {START_TASK} through {END_TASK}...")
    
    count = 0
    for i in range(START_TASK, END_TASK + 1):
        task_name = f"demo_task_full_{i:03d}"
        task_path = os.path.join(DEMO_ROOT, task_name)
        
        if not os.path.exists(task_path):
            print(f"Skipping {task_name} (Not found)")
            continue
            
        file_path = os.path.join(task_path, "head_camera_rgb_intrinsic_matrix.npy")
        
        # Overwrite with precise values
        np.save(file_path, precise_intrinsics)
        count += 1

    print(f"✅ Success! Updated {count} folders with correct camera calibration.")

if __name__ == "__main__":
    apply_precise_intrinsics()