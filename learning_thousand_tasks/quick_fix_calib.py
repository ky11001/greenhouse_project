import numpy as np
import os

# Path where the error said the file was missing
TARGET_PATH = "assets/T_WC_head_camera.npy"

def fix():
    # Create a standard 4x4 Identity Matrix
    # [[1, 0, 0, 0],
    #  [0, 1, 0, 0],
    #  [0, 0, 1, 0],
    #  [0, 0, 0, 1]]
    identity_matrix = np.eye(4)
    
    # Save it exactly where the script looks for it
    np.save(TARGET_PATH, identity_matrix)
    print(f"✅ Created placeholder calibration file: {TARGET_PATH}")

if __name__ == "__main__":
    fix()