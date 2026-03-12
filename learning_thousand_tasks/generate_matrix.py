import numpy as np
import os

# CONFIG
TASK_PATH = "/home/sci-lab/Documents/greenhouse_project/learning_thousand_tasks/assets/demonstrations/demo_task_full_006"

def save_custom_intrinsics():
    # Built from your actual RealSense hardware values
    intrinsics = np.array([
        [386.02066040,   0.0,          321.55554199],
        [  0.0,          385.41479492, 240.14253235],
        [  0.0,          0.0,          1.0         ]
    ], dtype=np.float64)

    save_path = os.path.join(TASK_PATH, "head_camera_rgb_intrinsic_matrix.npy")
    np.save(save_path, intrinsics)
    print(f"✓ Saved precise intrinsic matrix to {save_path}")

if __name__ == "__main__":
    save_custom_intrinsics() 