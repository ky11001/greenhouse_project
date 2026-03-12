import rospy
import numpy as np
import cv2
import os
import h5py
import argparse
import intera_interface
from intera_core_msgs.msg import EndpointState
from intera_interface import Gripper
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
from pynput import keyboard
from tf.transformations import quaternion_matrix

class DiagnosticRecorder:
    def __init__(self, task_name, do_annotate=False):
        self.task_name = task_name
        self.do_annotate = do_annotate
        self.bridge = CvBridge()
        self.limb = intera_interface.Limb('right')

        # --- CONFIGURATION ---
        self.save_root = os.path.expanduser("../greenhouse_data")
        if not os.path.exists(self.save_root): os.makedirs(self.save_root)

        self.intrinsics_matrix = np.array([
            [386.02066040,   0.0,          321.55554199],
            [  0.0,          385.41479492, 240.14253235],
            [  0.0,          0.0,          1.0         ]
        ], dtype=np.float64)

        self.joint_names = ['right_j0', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 'right_j5', 'right_j6']
        self.home_joints = [0, -1.57, 0.0, 1.57, -1.57, 0.0, 0.0]

        # Buffers
        self.img_buffer = []
        self.depth_buffer = []
        self.qpos_buffer = []
        self.twist_buffer = []
        self.pose_buffer = []
        
        # State Variables
        self.is_recording = False
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_twist = None
        self.latest_pose = None
        self.latest_joints = {} 
        self.ws_rgb = None

        # --- SUBSCRIBERS ---
        # 1. Camera
        self.rgb_topic = "/camera/color/image_raw"
        rospy.Subscriber(self.rgb_topic, Image, self.rgb_cb)
        rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_cb)
        
        # 2. Robot Endpoint
        self.endpoint_topic = "/robot/limb/right/endpoint_state"
        rospy.Subscriber(self.endpoint_topic, EndpointState, self.endpoint_cb)
        
        # 3. Robot Joints
        self.joint_topic = "/robot/joint_states"
        rospy.Subscriber(self.joint_topic, JointState, self.joint_cb)

        # Timer: 15Hz
        self.freq = 15.0
        self.timer = rospy.Timer(rospy.Duration(1.0/self.freq), self.recording_step)
        self.debug_timer = 0

        print(f"\n[READY] Task: {task_name}")
        print("-> PRESS SPACE: Start Recording")
        print("-> PRESS SPACE AGAIN: Stop & Save")
        print("-> ESC: Exit")

    def rgb_cb(self, msg):
        try: self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        except: pass

    def depth_cb(self, msg):
        try: self.latest_depth = self.bridge.imgmsg_to_cv2(msg, "16UC1")
        except: pass

    def endpoint_cb(self, msg):
        try:
            v = msg.twist.linear
            w = msg.twist.angular
            self.latest_twist = np.array([v.x, v.y, v.z, w.x, w.y, w.z])
            p = msg.pose.position
            q = msg.pose.orientation
            self.latest_pose = np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w])
        except: pass

    def joint_cb(self, msg):
        try:
            for i, name in enumerate(msg.name):
                self.latest_joints[name] = msg.position[i]
        except: pass

    def get_ordered_qpos(self):
        if len(self.latest_joints) < 7: return None
        try:
            return [self.latest_joints[j] for j in self.joint_names]
        except KeyError:
            return None

    def recording_step(self, event):
        if self.is_recording:
            qpos = self.get_ordered_qpos()
            
            # --- DEBUG BLOCK ---
            # Every 1 second (15 frames), print what is missing
            self.debug_timer += 1
            if self.debug_timer > 15:
                self.debug_timer = 0
                missing = []
                if self.latest_rgb is None: missing.append(f"CAMERA ({self.rgb_topic})")
                if self.latest_twist is None: missing.append(f"ROBOT ENDPOINT ({self.endpoint_topic})")
                if qpos is None: missing.append(f"ROBOT JOINTS ({self.joint_topic})")
                
                if len(missing) > 0:
                    print(f"[WAITING FOR]: {' + '.join(missing)}")
                else:
                    print(".", end="", flush=True) # Print dots when working
            # -------------------

            if (self.latest_rgb is not None and 
                self.latest_twist is not None and 
                qpos is not None):
                
                self.img_buffer.append(self.latest_rgb)
                if self.latest_depth is not None:
                    self.depth_buffer.append(self.latest_depth)
                self.qpos_buffer.append(qpos)
                self.twist_buffer.append(self.latest_twist)
                self.pose_buffer.append(self.latest_pose)

    def save_data(self):
        idx = 0
        while os.path.exists(os.path.join(self.save_root, f"{self.task_name}_{idx:03d}")):
            idx += 1
        folder_path = os.path.join(self.save_root, f"{self.task_name}_{idx:03d}")
        os.makedirs(folder_path)

        length = len(self.img_buffer)
        if length == 0:
            print("\nFAILURE: No data captured.")
            print("Check the [WAITING FOR] messages above to see what is missing.")
            return

        print(f"\nSaving {length} frames to: {folder_path}")

        rgb_arr = np.array(self.img_buffer)
        depth_arr = np.array(self.depth_buffer)
        full_action = np.array(self.twist_buffer)
        pose_arr = np.array(self.pose_buffer)
        qpos_joints = np.array(self.qpos_buffer)

        if self.ws_rgb is not None:
            cv2.imwrite(os.path.join(folder_path, "head_camera_ws_rgb.png"), cv2.cvtColor(self.ws_rgb, cv2.COLOR_RGB2BGR))

        np.save(os.path.join(folder_path, "head_camera_rgb_intrinsic_matrix.npy"), self.intrinsics_matrix)
        np.save(os.path.join(folder_path, "demo_eef_twists.npy"), full_action)
        np.save(os.path.join(folder_path, "head_camera_rgb.npy"), rgb_arr)
        if len(depth_arr) > 0:
            np.save(os.path.join(folder_path, "head_camera_depth.npy"), depth_arr)
            
        with open(os.path.join(folder_path, "task_name.txt"), "w") as f:
            f.write(self.task_name)

        min_z_idx = np.argmin(pose_arr[:, 2])
        bn_pose = pose_arr[min_z_idx]
        bn_matrix = quaternion_matrix(bn_pose[3:]) 
        bn_matrix[0, 3] = bn_pose[0]
        bn_matrix[1, 3] = bn_pose[1]
        bn_matrix[2, 3] = bn_pose[2]
        np.save(os.path.join(folder_path, "bottleneck_pose.npy"), bn_matrix)

        h5_path = os.path.join(folder_path, "episode_data.hdf5")
        with h5py.File(h5_path, 'w') as root:
            root.create_dataset('action', data=full_action)
            obs = root.create_group('observations')
            obs.create_dataset('qpos', data=qpos_joints)
            obs.create_dataset('images/cam_high', data=rgb_arr)
            obs.create_dataset('eef_pose', data=pose_arr)

        print("✅ MT3 & ACT Data Saved.")
        
        if self.do_annotate:
            self.perform_annotation(folder_path, self.ws_rgb)

    def perform_annotation(self, folder_path, rgb_img):
        print("\n--- ANNOTATION ---")
        if rgb_img is None: return
        bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        roi = cv2.selectROI("Annotate", bgr, showCrosshair=True)
        cv2.destroyWindow("Annotate")
        
        x, y, w, h = roi
        if w > 0:
            mask = np.zeros(bgr.shape[:2], dtype=bool)
            mask[y:y+h, x:x+w] = True
            np.save(os.path.join(folder_path, "head_camera_ws_segmap.npy"), mask)

    def reset_buffers(self):
        self.img_buffer = []
        self.depth_buffer = []
        self.qpos_buffer = []
        self.twist_buffer = []
        self.pose_buffer = []

    def on_press(self, key):
        if key == keyboard.Key.space:
            if not self.is_recording:
                self.reset_buffers()
                if self.latest_rgb is not None:
                    self.ws_rgb = self.latest_rgb.copy()
                self.is_recording = True
                print("● RECORDING (Diagnostics Enabled)...")
            else:
                self.is_recording = False
                print("■ STOPPED. Saving...")
                self.save_data()
                self.limb.move_to_joint_positions(dict(zip(self.joint_names, self.home_joints)))

        if key == keyboard.Key.esc: return False

if __name__ == '__main__':
    rospy.init_node('diagnostic_recorder')
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotate", action="store_true", help="Draw segmentation mask")
    args = parser.parse_args()
    
    rec = DiagnosticRecorder("demo_task", do_annotate=args.annotate)
    with keyboard.Listener(on_press=rec.on_press) as l: l.join()