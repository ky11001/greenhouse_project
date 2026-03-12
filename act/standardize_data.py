import os
import h5py
import numpy as np
from tqdm import tqdm
import shutil

# --- CONFIGURATION ---
INPUT_DIR = '/home/sci-lab/Documents/greenhouse_project/act_data'
OUTPUT_DIR = '/home/sci-lab/Documents/greenhouse_project/act_data_padded'
# If you want to force a specific length (e.g. 400), set this. 
# If None, it will automatically find the longest episode in your data.
FORCED_MAX_LEN = None 
# ---------------------

def standardize_dataset():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.hdf5')]
    files.sort()
    
    # Step 1: Find the maximum length
    max_len = 0
    if FORCED_MAX_LEN:
        max_len = FORCED_MAX_LEN
        print(f"Using forced max length: {max_len}")
    else:
        print("Scanning files to find max episode length...")
        for f_name in tqdm(files):
            path = os.path.join(INPUT_DIR, f_name)
            with h5py.File(path, 'r') as root:
                l = root['/action'].shape[0]
                if l > max_len:
                    max_len = l
        print(f"Maximum episode length found: {max_len}")

    # Step 2: Pad and Save
    print(f"Padding all episodes to {max_len} frames...")
    
    for f_name in tqdm(files):
        src_path = os.path.join(INPUT_DIR, f_name)
        dst_path = os.path.join(OUTPUT_DIR, f_name)

        with h5py.File(src_path, 'r') as src:
            # Read original data
            qpos = src['/observations/qpos'][()]
            qvel = src['/observations/qvel'][()]
            action = src['/action'][()]
            
            # Read attributes
            attrs = dict(src.attrs)
            
            # Read images
            images = {}
            for cam_name in src['/observations/images'].keys():
                images[cam_name] = src[f'/observations/images/{cam_name}'][()]

            # Calculate padding needed
            curr_len = qpos.shape[0]
            pad_len = max_len - curr_len

            if pad_len < 0:
                print(f"Warning: {f_name} is longer than max_len ({curr_len} > {max_len}). Truncating.")
                # Truncate logic if needed, but usually we just pad up
                qpos = qpos[:max_len]
                qvel = qvel[:max_len]
                action = action[:max_len]
                for k in images:
                    images[k] = images[k][:max_len]
            elif pad_len > 0:
                # --- PADDING LOGIC ---
                # 1. Action: Repeat the last action (hold position)
                last_action = action[-1]
                pad_action = np.tile(last_action, (pad_len, 1))
                action = np.concatenate([action, pad_action], axis=0)

                # 2. QPos: Repeat the last position
                last_qpos = qpos[-1]
                pad_qpos = np.tile(last_qpos, (pad_len, 1))
                qpos = np.concatenate([qpos, pad_qpos], axis=0)

                # 3. QVel: Fill with ZEROS (stopped)
                pad_qvel = np.zeros((pad_len, qvel.shape[1]))
                qvel = np.concatenate([qvel, pad_qvel], axis=0)

                # 4. Images: Repeat the last image
                for k in images:
                    last_img = images[k][-1]
                    # Expand dims to match (pad_len, h, w, c)
                    pad_img = np.tile(last_img[None, ...], (pad_len, 1, 1, 1))
                    images[k] = np.concatenate([images[k], pad_img], axis=0)

            # Save to new file
            with h5py.File(dst_path, 'w') as dst:
                # Save attributes (sim, etc)
                for k, v in attrs.items():
                    dst.attrs[k] = v
                
                # Save datasets
                dst.create_dataset('/observations/qpos', data=qpos)
                dst.create_dataset('/observations/qvel', data=qvel)
                dst.create_dataset('/action', data=action)
                
                for k, v in images.items():
                    dst.create_dataset(f'/observations/images/{k}', data=v, compression='gzip')

    print(f"Done! Processed data saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    standardize_dataset()