import rospy
import numpy as np
import cv2
import os
import argparse
import intera_interface
from intera_interface import Gripper
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from pynput import keyboard
from PIL import Image as PILImage

class FullStateRecorder:
    def __init__(self, task_name, do_annotate=False):
        self.task_name = task_name
        self.do_annotate = do_annotate
        self.bridge = CvBridge()
        self.limb = intera_interface.Limb('right')
        
        # --- CONFIGURATION ---
        self.save_root = os.path.expanduser("~/Documents/greenhouse_project/learning_thousand_tasks/assets/demonstrations/")
        self.home_joints = [0, -1.57, 0.0, 1.57, -1.57, 0.0, 0.0]
        self.joint_names = ['right_j0', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 'right_j5', 'right_j6']
        
        # Precise RealSense Intrinsics (from your hardware)
        self.intrinsics_matrix = np.array([
            [386.02066040,   0.0,          321.55554199],
            [  0.0,          385.41479492, 240.14253235],
            [  0.0,          0.0,          1.0         ]
        ], dtype=np.float64)

        # Initialize Gripper
        try:
            self.gripper = Gripper('right_hand')
            print("[INFO] Gripper initialized.")
        except:
            self.gripper = None
            print("[WARNING] No gripper detected.")

        self.rgb_topic = "/camera/color/image_raw"
        self.depth_topic = "/camera/aligned_depth_to_color/image_raw"
        
        self.is_recording = False
        self.reset_buffers()
        self.has_received_rgb = False

        print(f"\n[WAITING] Connecting to RealSense topics...")
        rospy.Subscriber(self.rgb_topic, Image, self.rgb_cb)
        rospy.Subscriber(self.depth_topic, Image, self.depth_cb)
        
        # Connection check
        rospy.sleep(2.0)
        if not self.has_received_rgb:
             self.print_realsense_help()

        print(f"\n[READY] Task Base Name: {task_name}")
        print(f"[MODE] Annotation is {'ON' if self.do_annotate else 'OFF'}")
        print("-> PRESS SPACEBAR: Start Recording")
        print("-> PRESS SPACEBAR AGAIN: Stop, Save & Move Home")
        print("-> ESC: Exit")

    def print_realsense_help(self):
        print("\n" + "!"*60)
        print("CRITICAL ERROR: REALSENSE NOT DETECTED")
        print("1. Check USB connection (3.0 preferred).")
        print("2. Run this in a new terminal:")
        print("   roslaunch realsense2_camera rs_camera.launch align_depth:=true")
        print("!"*60 + "\n")

    def reset_buffers(self):
        self.rs_rgb = []
        self.rs_depth = []
        self.full_state = []

    def rgb_cb(self, msg):
        self.has_received_rgb = True
        if self.is_recording:
            try:
                # Store frame (convert BGR to RGB for preprocessor)
                frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                self.rs_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # Robot Data
                j_angles = self.limb.joint_angles()
                # j_vels = self.limb.joint_velocities() # Optional
                # j_efforts = self.limb.joint_efforts() # Optional
                curr_angles = [j_angles[j] for j in self.joint_names] 
                
                p = self.limb.endpoint_pose()
                cartesian = [p['position'].x, p['position'].y, p['position'].z,
                             p['orientation'].x, p['orientation'].y, p['orientation'].z, p['orientation'].w] 
                
                g_pos = self.gripper.get_position() if self.gripper else 0.0
                
                # Save simplified state for robot_states.npy [Joints(7) + Pose(7) + Gripper(1)]
                # Total: 15 dims
                state_vector = curr_angles + cartesian + [g_pos]
                self.full_state.append(state_vector)
            except Exception as e:
                print(f"Error: {e}")

    def depth_cb(self, msg):
        if self.is_recording:
            try:
                self.rs_depth.append(self.bridge.imgmsg_to_cv2(msg, "16UC1"))
            except Exception as e: pass

    def get_next_folder_path(self):
        """Finds the next available folder index to avoid overwriting."""
        idx = 1
        while True:
            folder_name = f"{self.task_name}_{idx:03d}"
            full_path = os.path.join(self.save_root, folder_name)
            if not os.path.exists(full_path):
                return full_path
            idx += 1

    def perform_annotation(self, folder_path, rgb_frame):
        """Opens a window to draw the segmentation mask immediately."""
        print("\n--- ANNOTATION MODE ---")
        print("Draw a box around the object and press ENTER. Press 'c' to cancel.")
        
        # Convert back to BGR for OpenCV display
        bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        roi = cv2.selectROI("Annotate Object", bgr, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Annotate Object")
        
        x, y, w, h = roi
        if w > 0 and h > 0:
            mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
            mask[y:y+h, x:x+w] = 1 # 1 = Object, 0 = Background
            
            save_path = os.path.join(folder_path, "head_camera_ws_segmap.npy")
            np.save(save_path, mask)
            print(f"✅ Saved mask to {save_path}")
        else:
            print("⚠️ No mask selected. You will need to generate this later.")

    def save_data(self):
        folder = self.get_next_folder_path()
        os.makedirs(folder, exist_ok=True)
        
        length = min(len(self.rs_rgb), len(self.rs_depth), len(self.full_state))
        if length == 0:
            print("FAILURE: No data captured.")
            return

        print(f"Saving {length} frames to: {folder}...")

        # --- 1. Save Core Video Data ---
        # Note: Preprocessor expects 'head_camera_depth.npy', not 'depth_to_rgb'
        np.save(os.path.join(folder, "head_camera_rgb.npy"), np.array(self.rs_rgb[:length]))
        np.save(os.path.join(folder, "head_camera_depth.npy"), np.array(self.rs_depth[:length]))
        
        # --- 2. Save Robot States (Master File) ---
        state_stack = np.array(self.full_state[:length])
        np.save(os.path.join(folder, "robot_states.npy"), state_stack)

        # --- 3. Save Bottleneck (Smart Min-Z Logic) ---
        # Indices 7,8,9 are x,y,z in our 15-dim vector
        z_values = state_stack[:, 9] 
        bottleneck_idx = np.argmin(z_values) # Find lowest point
        
        # Indices 7-14 are the 7D Pose [x,y,z,qx,qy,qz,qw]
        bottleneck_pose = state_stack[bottleneck_idx, 7:14]
        
        np.save(os.path.join(folder, "bottleneck_posevec.npy"), bottleneck_pose)
        np.save(os.path.join(folder, "bottleneck_pose.npy"), bottleneck_pose) # Duplicate for safety

        # --- 4. Save Intrinsics ---
        np.save(os.path.join(folder, "head_camera_rgb_intrinsic_matrix.npy"), self.intrinsics_matrix)

        # --- 5. Save Workspace Image (For visualization/debugging) ---
        ws_rgb = cv2.cvtColor(self.rs_rgb[0], cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(folder, "head_camera_ws_rgb.png"), ws_rgb)

        print(f"✅ Data Saved.")

        # --- 6. Optional Annotation ---
        if self.do_annotate:
            self.perform_annotation(folder, self.rs_rgb[0])

    def on_press(self, key):
        if key == keyboard.Key.space:
            if not self.is_recording:
                self.reset_buffers()
                self.is_recording = True
                print("● RECORDING STARTED...")
            else:
                self.is_recording = False
                print("■ STOPPED. Processing...")
                self.save_data()
                # Move home after save
                self.limb.move_to_joint_positions(dict(zip(self.joint_names, self.home_joints)))
        if key == keyboard.Key.esc: return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotate", action="store_true", help="Draw segmentation mask after recording")
    args = parser.parse_args()

    rospy.init_node('full_pipeline_recorder')
    
    # NOTE: Change 'demo_task_full' to whatever base name you want
    rec = FullStateRecorder("demo_task_full", do_annotate=args.annotate) 
    
    with keyboard.Listener(on_press=rec.on_press) as l: l.join() 