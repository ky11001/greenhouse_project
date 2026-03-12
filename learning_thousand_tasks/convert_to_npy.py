import numpy as np
import cv2
import glob
import os

# --- CONFIG ---
TASK_NAME = "demo_task_full_004" # <--- UPDATE THIS
DEMO_ROOT = "assets/demonstrations/"

def convert():
    path = os.path.join(DEMO_ROOT, TASK_NAME)
    print(f"Processing {path}...")
    
    # 1. RGB Images
    files = sorted(glob.glob(os.path.join(path, "color_*.jpg")), 
                   key=lambda x: int(x.split('_')[-1].split('.')[0]))
    if not files:
        print("Error: No images found!")
        return

    # Convert BGR (OpenCV) -> RGB (Numpy)
    stack = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in files]
    np.save(os.path.join(path, "head_camera_rgb.npy"), np.array(stack))
    
    # 2. Depth Images
    dfiles = sorted(glob.glob(os.path.join(path, "depth_*.npy")), 
                   key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # Convert mm (uint16) -> Meters (float32)
    dstack = [np.load(f).astype(np.float32) / 1000.0 for f in dfiles]
    np.save(os.path.join(path, "head_camera_depth.npy"), np.array(dstack))
    
    print(f"Done! Created head_camera_rgb.npy & depth.npy with {len(stack)} frames.")

if __name__ == "__main__":
    convert()