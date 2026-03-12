import cv2
import os
import glob
import re

# CONFIGURATION
TASK_NAME = "demo_task_full_001"  # The specific folder you want to convert
DATA_ROOT = "assets/demonstrations/"
FPS = 15  # Matches your RealSense recording speed

def natural_sort_key(s):
    """Sorts strings with numbers naturally (e.g., color_2.jpg comes before color_10.jpg)"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def make_video():
    folder_path = os.path.join(DATA_ROOT, TASK_NAME)
    video_name = os.path.join(folder_path, f"{TASK_NAME}_video.mp4")
    
    # Get all color images
    images = glob.glob(os.path.join(folder_path, "color_*.jpg"))
    images.sort(key=natural_sort_key) # Ensure strict 0, 1, 2, 3 order

    if not images:
        print(f"No images found in {folder_path}")
        return

    # Read first image to get dimensions
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    # 'mp4v' is a standard MP4 codec. 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(video_name, fourcc, FPS, (width, height))

    print(f"Creating video: {video_name}")
    print(f"Found {len(images)} frames. Processing...")

    for img_path in images:
        frame = cv2.imread(img_path)
        video.write(frame)

    video.release()
    print("Done! Video saved.")

if __name__ == "__main__":
    make_video()