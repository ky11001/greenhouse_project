#!/usr/bin/env python3

import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt


def print_hdf5_structure(file_path):
    print("\n--- HDF5 STRUCTURE ---")
    with h5py.File(file_path, 'r') as f:
        def visitor(name, obj):
            print(name)
        f.visititems(visitor)
    print("----------------------\n")


def load_data(file_path):
    with h5py.File(file_path, 'r') as f:
        action = np.array(f['action'])
        qpos = np.array(f['observations/qpos'])
        eef_pose = np.array(f['observations/eef_pose'])
        images = np.array(f['observations/images/cam_high'])

    return action, qpos, eef_pose, images


def main():

    if len(sys.argv) < 2:
        print("Usage: python plot_episode.py /path/to/episode_data.hdf5")
        sys.exit(1)

    file_path = sys.argv[1]

    # 1️⃣ Print structure
    print_hdf5_structure(file_path)

    # 2️⃣ Load datasets
    action, qpos, eef_pose, images = load_data(file_path)

    print("Loaded Shapes:")
    print("  action:", action.shape)
    print("  qpos:", qpos.shape)
    print("  eef_pose:", eef_pose.shape)
    print("  images:", images.shape)

    timesteps = np.arange(qpos.shape[0])

    # 3️⃣ Create plots
    fig, axs = plt.subplots(4, 1, figsize=(12, 16))

    # --- Joint Positions ---
    for i in range(qpos.shape[1]):
        axs[0].plot(timesteps, qpos[:, i], label=f'joint_{i}')
    axs[0].set_title("Joint Positions (qpos)")
    axs[0].set_xlabel("Timestep")
    axs[0].set_ylabel("Radians")
    axs[0].legend()
    axs[0].grid(True)

    # --- End Effector Z ---
    z_vals = eef_pose[:, 2]
    axs[1].plot(timesteps, z_vals, color='black')
    axs[1].set_title("End Effector Z Height")
    axs[1].set_xlabel("Timestep")
    axs[1].set_ylabel("Z (meters)")
    axs[1].grid(True)

    # --- Action (Twist) ---
    for i in range(action.shape[1]):
        axs[2].plot(timesteps, action[:, i], label=f'action_{i}')
    axs[2].set_title("EEF Twist (Linear + Angular)")
    axs[2].set_xlabel("Timestep")
    axs[2].legend()
    axs[2].grid(True)

    # --- Show First Image Frame ---
    axs[3].imshow(images[0])
    axs[3].set_title("First RGB Frame")
    axs[3].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
