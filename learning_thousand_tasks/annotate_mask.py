import cv2
import numpy as np
import os

TASK_NAME = "demo_task_full_006" # UPDATE THIS
DEMO_ROOT = "assets/demonstrations/"

def annotate():
    path = os.path.join(DEMO_ROOT, TASK_NAME)
    rgb_path = os.path.join(path, "head_camera_rgb.npy")
    
    if not os.path.exists(rgb_path):
        print("Run convert_to_npy.py first!")
        return

    # Load first frame
    rgb = np.load(rgb_path)[0]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    
    print("INSTRUCTIONS: Draw box, press ENTER. Press 'c' to cancel.")
    roi = cv2.selectROI("Annotate Object", bgr, fromCenter=False)
    cv2.destroyAllWindows()
    
    x, y, w, h = roi
    mask = np.zeros(bgr.shape[:2], dtype=bool)
    
    if w > 0 and h > 0:
        mask[y:y+h, x:x+w] = True
        np.save(os.path.join(path, "head_camera_ws_segmap.npy"), mask)
        print("Success! Saved head_camera_ws_segmap.npy")
    else:
        print("No selection made.")

if __name__ == "__main__":
    annotate()