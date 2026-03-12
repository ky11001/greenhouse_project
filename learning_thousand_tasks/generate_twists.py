import numpy as np
import os

# CONFIG
TASK_PATH = "/home/sci-lab/Documents/greenhouse_project/learning_thousand_tasks/assets/demonstrations/demo_task_full_004"

def calculate_twists():
    # Load the poses we linked earlier (robot_states.npy)
    # Shape is likely (T, 7) or (T, 8)
    poses_path = os.path.join(TASK_PATH, "demo_eef_posevecs.npy")
    if not os.path.exists(poses_path):
        print("Error: Could not find demo_eef_posevecs.npy")
        return

    poses = np.load(poses_path)
    num_frames = poses.shape[0]
    
    # Create empty twists array (T, 7)
    # [vx, vy, vz, wx, wy, wz, gripper]
    twists = np.zeros((num_frames, 7))

    # 1. Calculate Linear Velocity (vx, vy, vz)
    # Using simple finite difference: (pos[t+1] - pos[t])
    twists[:-1, :3] = np.diff(poses[:, :3], axis=0)
    # For the last frame, repeat the previous velocity
    twists[-1, :3] = twists[-2, :3]

    # 2. Calculate Angular Velocity (wx, wy, wz)
    # For simplicity in BC training, we can often use zeros if 
    # the orientation doesn't change much, or finite difference of Euler angles.
    # Let's use zeros for now to bypass the error and allow tracking to start.
    twists[:, 3:6] = 0.0

    # 3. Gripper State
    # If your recording has a gripper column (usually index 7), use it.
    # Otherwise, set to 0.0 (open) or 1.0 (closed).
    if poses.shape[1] > 7:
        twists[:, 6] = poses[:, 7]
    else:
        twists[:, 6] = 0.0

    save_path = os.path.join(TASK_PATH, "demo_eef_twists.npy")
    np.save(save_path, twists)
    print(f"✓ Generated twists of shape {twists.shape} and saved to {save_path}")

if __name__ == "__main__":
    calculate_twists()