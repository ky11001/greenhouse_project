import numpy as np
from PIL import Image
import os

# CONFIG
DEMO_ROOT = "/home/sci-lab/Documents/greenhouse_project/learning_thousand_tasks/assets/demonstrations"
START_TASK = 32
END_TASK = 41

def fix_ws_depth():
    print("--- Generating Missing Workspace Depth Images ---")
    count = 0
    
    for i in range(START_TASK, END_TASK + 1):
        task_name = f"demo_task_full_{i:03d}"
        task_path = os.path.join(DEMO_ROOT, task_name)
        
        if not os.path.exists(task_path):
            continue

        # Source: The full video array we just renamed
        npy_path = os.path.join(task_path, "head_camera_depth_to_rgb.npy")
        # Target: The single image file the preprocessor wants
        target_png = os.path.join(task_path, "head_camera_ws_depth_to_rgb.png")
        
        # Check if we need to generate it
        if not os.path.exists(target_png):
            if os.path.exists(npy_path):
                try:
                    # Load the full video data
                    depth_video = np.load(npy_path)
                    
                    if len(depth_video) > 0:
                        # Grab the first frame (Workspace State)
                        first_frame = depth_video[0]
                        
                        # Ensure it's the correct format for PIL (uint16 is standard for depth)
                        if first_frame.dtype != np.uint16:
                             first_frame = first_frame.astype(np.uint16)
                             
                        # Save as 16-bit PNG
                        img = Image.fromarray(first_frame, mode='I;16')
                        img.save(target_png)
                        
                        print(f"✅ Generated PNG for {task_name}")
                        count += 1
                    else:
                        print(f"⚠️  {task_name} depth file is empty!")
                except Exception as e:
                    print(f"❌ Error processing {task_name}: {e}")
            else:
                print(f"⚠️  Skipping {task_name} (Source .npy not found)")
        else:
            # File exists, moving on
            pass

    print(f"\nSummary: Generated {count} missing workspace depth images.")

if __name__ == "__main__":
    fix_ws_depth()