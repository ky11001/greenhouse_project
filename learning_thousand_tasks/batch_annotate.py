import cv2
import numpy as np
import os

# Configuration
DEMO_ROOT = "assets/demonstrations/"
START_TASK = 32
END_TASK = 41

def batch_annotate():
    print(f"Starting batch annotation for tasks {START_TASK} to {END_TASK}...")
    
    for i in range(START_TASK, END_TASK + 1):
        task_name = f"demo_task_full_{i:03d}"
        path = os.path.join(DEMO_ROOT, task_name)
        rgb_path = os.path.join(path, "head_camera_rgb.npy")
        segmap_path = os.path.join(path, "head_camera_ws_segmap.npy")

        # 1. Validation Checks
        if not os.path.exists(path):
            print(f"Skipping {task_name}: Folder not found.")
            continue
            
        if not os.path.exists(rgb_path):
            print(f"Skipping {task_name}: head_camera_rgb.npy missing.")
            continue
            
        # 2. Check if already done
        if os.path.exists(segmap_path):
            print(f"Task {task_name} already has a segmap.")
            choice = input("  [S]kip or [R]edo? (s/r): ").lower()
            if choice != 'r':
                continue

        # 3. Load Image
        try:
            # Load the full video, but take only the first frame [0]
            rgb_video = np.load(rgb_path, mmap_mode='r') 
            first_frame_rgb = rgb_video[0]
            bgr = cv2.cvtColor(first_frame_rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Error loading {task_name}: {e}")
            continue

        # 4. User Interaction (ROI Selection)
        print(f"\n--- Annotating: {task_name} ---")
        print("INSTRUCTIONS:\n 1. Click & Drag a box around the object.\n 2. Press ENTER or SPACE to confirm.\n 3. Press 'c' to cancel (skip this task).")
        
        # Opens the window
        roi = cv2.selectROI(f"Annotate: {task_name}", bgr, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(f"Annotate: {task_name}") # Close window after selection
        
        # 5. Process Selection
        x, y, w, h = roi
        
        # If user pressed 'c' or selected nothing, w and h will be 0
        if w > 0 and h > 0:
            # Create mask (0 = background, 1 = object)
            mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
            mask[y:y+h, x:x+w] = 1 
            
            np.save(segmap_path, mask)
            print(f"✅ Saved mask for {task_name}")
        else:
            print(f"⚠️ Skipped {task_name} (No selection made)")

    print("\nBatch annotation complete!")

if __name__ == "__main__":
    batch_annotate()