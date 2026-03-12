import h5py
import os
import numpy as np

# Update this to your actual dataset path
dataset_dir = '../act/data/greenhouse_tasks'
file_path = os.path.join(dataset_dir, 'episode_0.hdf5')

if not os.path.exists(file_path):
    print(f"Error: Could not find {file_path}")
    # List files in the directory to help you debug
    print("Files in directory:", os.listdir(dataset_dir) if os.path.exists(dataset_dir) else "Directory not found")
else:
    with h5py.File(file_path, 'r') as root:
        qpos = root['/observations/qpos'][()]
        qvel = root['/observations/qvel'][()]
        print(f"Success! Loaded episode_0")
        print(f"Joint Positions (qpos) shape: {qpos.shape}")
        print(f"Joint Velocities (qvel) shape: {qvel.shape}")