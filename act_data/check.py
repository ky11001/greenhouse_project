import numpy as np
import matplotlib.pyplot as plt
import h5py  # <--- REQUIRED for .hdf5 files

def check_data_health(file_path):
    print(f"--- INSPECTING: {file_path} ---")
    
    # 1. LOAD DATA CORRECTLY
    try:
        with h5py.File(file_path, 'r') as f:
            # Print keys to help you debug file structure
            print("File Keys:", list(f.keys()))
            
            # TRY TO FIND QPOS DATA
            # Standard ACT format often hides qpos inside 'observations'
            if 'qpos' in f.keys():
                qpos_data = f['qpos'][:]
            elif 'observations' in f.keys() and 'qpos' in f['observations'].keys():
                qpos_data = f['observations']['qpos'][:]
            elif 'action' in f.keys():
                # Sometimes we want to check 'action' instead of 'qpos'
                print("Note: Using 'action' as proxy for qpos.")
                qpos_data = f['action'][:]
            else:
                print("❌ ERROR: Could not find 'qpos' dataset. Check the keys printed above.")
                return

    except Exception as e:
        print(f"❌ Failed to open file: {e}")
        return

    # Ensure shape is (Time, Joints)
    print(f"Data Shape: {qpos_data.shape}")
    
    # 2. RUN HEALTH CHECK
    # Calculate velocity (difference between steps)
    velocities = np.diff(qpos_data, axis=0)
    avg_velocity = np.mean(np.abs(velocities))
    
    # Threshold for "stopped" (adjust based on noise, 0.001 rad is roughly 0.05 degrees)
    is_stopped = np.all(np.abs(velocities) < 0.002, axis=1)
    percent_idle = np.sum(is_stopped) / len(is_stopped) * 100
    
    print(f"Avg Velocity per step: {avg_velocity:.5f}")
    print(f"Percentage of IDLE frames: {percent_idle:.1f}%")
    
    if percent_idle > 20:
        print("⚠️  WARNING: High Idle Time (>20%). Model will learn to 'pause'. Trim start/end of data.")
    elif percent_idle < 5:
        print("✅  Idle time looks good (low).")
    else:
        print("ℹ️  Idle time is moderate.")

    # 3. Check Range/Normalization Prep
    print(f"\nJoint Ranges (Max - Min):")
    ranges = qpos_data.max(axis=0) - qpos_data.min(axis=0)
    print(np.round(ranges, 4))
    
    # Identify frozen joints
    frozen_joints = np.where(ranges < 0.01)[0]
    if len(frozen_joints) > 0:
        print(f"⚠️  WARNING: Joints {frozen_joints} basically never move.")
        print("    -> Set their std_dev to 1.0 in your stats file to avoid explosion.")
        
    # 4. Visualization
    plt.figure(figsize=(10, 5))
    plt.plot(qpos_data)
    plt.title(f"Joint Trajectories: {file_path}")
    plt.xlabel("Time Steps")
    plt.ylabel("Joint Angles (Rad)")
    plt.grid(True, alpha=0.3)
    plt.show()

# --- USAGE ---
# Replace with your actual file path
check_data_health('episode_0.hdf5')