import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# --- CONFIGURATION ---
TASK_FOLDER = "demo_task_000" 

def save_segmap_preview():
    rgb_path = os.path.join(TASK_FOLDER, "head_camera_ws_rgb.png")
    seg_path = os.path.join(TASK_FOLDER, "head_camera_ws_segmap.npy")

    if not os.path.exists(rgb_path) or not os.path.exists(seg_path):
        print(f"Error: Files not found in {TASK_FOLDER}")
        return

    # 1. Load the data
    # head_camera_ws_rgb.png is (720, 1280, 3)
    rgb_img = cv2.imread(rgb_path)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    
    # head_camera_ws_segmap.npy is (720, 1280) bool
    segmap = np.load(seg_path)

    # 2. Create the overlay
    overlay = rgb_img.copy()
    overlay[segmap] = [0, 255, 0] # Turn the masked area green

    # 3. Save to file instead of showing
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    
    axs[0].imshow(rgb_img)
    axs[0].set_title("Original Workspace RGB")
    axs[0].axis('off')

    axs[1].imshow(rgb_img)
    axs[1].imshow(overlay, alpha=0.5) 
    axs[1].set_title("Segmap Overlay (Target in Green)")
    axs[1].axis('off')

    save_name = "segmap_check.png"
    plt.savefig(save_name)
    print(f"✅ Preview saved to {save_name}. You can open this file to check the mask.")

if __name__ == "__main__":
    save_segmap_preview()