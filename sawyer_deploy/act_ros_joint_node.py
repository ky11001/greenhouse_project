#!/usr/bin/env python3
import sys
from time import time

# --- HACK TO SILENCE ARGPARSE ---
sys.argv = ['act_ros_node.py']

import rospy
import torch
import numpy as np
import pickle
import cv2
import threading
from collections import deque
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg') # Forces Matplotlib to bypass Qt entirely
from matplotlib.animation import FuncAnimation

from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
import torchvision.transforms as transforms

# import realsense2_camera

# --- IMPORTS FROM YOUR FOLDER ---
from policy import ACTPolicy 

# --- FIXED CONFIG CLASS (ROBUST VERSION) ---
class ModelArgs:
    def __init__(self, config_dict):
        self._config_dict = config_dict
        for k, v in config_dict.items():
            setattr(self, k, v)
            
    def items(self):
        return self._config_dict.items()
        
    def __getitem__(self, key):
        return self._config_dict[key]
    
    def __contains__(self, key):
        return key in self._config_dict

class ACTNode:
    def __init__(self):
        rospy.init_node('act_inference_node')
        
        # 1. CONFIGURATION
        config = {
            'lr': 1e-5, 
            'num_queries': 54,   
            'kl_weight': 10, 
            'state_dim': 7,
            'hidden_dim': 512, 
            'dim_feedforward': 3200, 
            'lr_backbone': 1e-5, 
            'backbone': 'resnet18', 
            'enc_layers': 4, 
            'dec_layers': 7, 
            'nheads': 8, 
            'camera_names': ['cam_high'], 
            'masks': False,
            'dilation': False,
            'dropout': 0.1,
            'pre_norm': False,
            'batch_size': 1,      
            'weight_decay': 1e-4, 
        }
        
        self.args = ModelArgs(config)
        self.ckpt_path = 'policy_best.ckpt' 
        self.stats_path = 'dataset_stats.pkl' 
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bridge = CvBridge()
        
        self.setup_model()
        
        self.pub = rospy.Publisher('/desired_joint_velocity', Float64MultiArray, queue_size=1)

        rospy.Subscriber('/camera/color/image_raw', Image, self.img_callback, queue_size=10) 
        rospy.Subscriber('/robot/joint_states', JointState, self.joint_callback, queue_size=10)

        self.curr_img = None
        self.curr_qpos = None
        self.raw_qpos = None 
        
        self.arm_joints = ['right_j0', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 'right_j5', 'right_j6']
        self.normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.control_rate = 10
        self.rate = rospy.Rate(self.control_rate) 
        
        # Temporal Ensembling variables
        self.chunk_history = deque(maxlen=self.args.num_queries)
        self.k = 0.01

        # Plotting variables
        self.time_history = []
        self.vel_history = []
        self.start_time = time()
        self.last_valid_velocity = np.zeros(7)


    def setup_model(self):
        rospy.loginfo(f"Loading Model from {self.ckpt_path}...")
        with open(self.stats_path, 'rb') as f:
            self.stats = pickle.load(f)
            
        self.policy = ACTPolicy(self.args)
        checkpoint = torch.load(self.ckpt_path, map_location=self.device)
        self.policy.load_state_dict(checkpoint)
        self.policy.to(self.device)
        self.policy.eval()
        rospy.loginfo("Model Loaded Successfully!")

    def img_callback(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv_img = cv2.resize(cv_img, (640, 480))
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            tens = torch.from_numpy(cv_img).permute(2, 0, 1).float() / 255.0
            self.curr_img = self.normalizer(tens).unsqueeze(0).unsqueeze(0).to(self.device)
        except Exception as e:
            print("Error in Image Callback:", e)

    def joint_callback(self, msg):
        try:
            val_dict = dict(zip(msg.name, msg.position))
            qpos = np.array([val_dict[j] for j in self.arm_joints])
            self.raw_qpos = qpos 
            qpos_norm = (qpos - self.stats['qpos_mean']) / self.stats['qpos_std']
            self.curr_qpos = torch.from_numpy(qpos_norm).float().unsqueeze(0).to(self.device)
        except KeyError:
            pass 

    def run(self):
        rospy.loginfo("ACT Node Running with Temporal Ensembling...")
        dt = 1.0 / self.control_rate

        while not rospy.is_shutdown():
            if self.curr_img is not None and self.curr_qpos is not None and self.raw_qpos is not None:
                with torch.inference_mode():
                    all_actions = self.policy(self.curr_qpos, self.curr_img)
                
                # 1. Temporal Ensembling
                chunk = all_actions.squeeze().cpu().numpy() 
                self.chunk_history.append(chunk)
                
                current_actions = []
                for i, past_chunk in enumerate(reversed(self.chunk_history)):
                    current_actions.append(past_chunk[i])
                current_actions = np.array(current_actions) 
                
                weights = np.exp(-self.k * np.arange(len(current_actions)))
                weights = weights / weights.sum() 
                raw_action = np.sum(current_actions * weights[:, np.newaxis], axis=0)
                
                # 2. Math & Publish
                target_qpos = (raw_action * self.stats['action_std']) + self.stats['action_mean']
                joint_velocities = (target_qpos - self.raw_qpos) / dt / 20
                # joint_velocities = np.clip(joint_velocities, -0.2, 0.2)
                
                msg = Float64MultiArray()
                msg.data = joint_velocities.tolist()
                self.pub.publish(msg) # Commented out so it doesn't move the real robot yet

                # 3. Log data for the live plot
                t_current = time() - self.start_time
                self.time_history.append(t_current)
                self.vel_history.append(joint_velocities.copy())
                print("Image and joint data received")

            else:
                rospy.loginfo_throttle(2.0, "Waiting for image and joint data...")
                
            
            self.rate.sleep()


if __name__ == '__main__':
    try:
        node = ACTNode()

        # Start the ROS loop in a background thread
        ros_thread = threading.Thread(target=node.run)
        ros_thread.daemon = True
        ros_thread.start()

        # Setup Live Plotting on the main thread
        fig, ax = plt.subplots(figsize=(10, 6))
        lines = [ax.plot([], [], label=f'j{i}')[0] for i in range(7)]
        ax.set_ylim(-1, 1)
        ax.legend(loc='upper right')
        ax.set_title("Neural Network Commanded Velocities (Smoothed)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Velocity (rad/s)")
        ax.grid(True)

        def update_plot(frame):
            if len(node.time_history) < 2:
                return lines

            # Keep only the last 100 points so the graph scrolls cleanly
            times = np.array(node.time_history[-100:])
            vels = np.array(node.vel_history[-100:])

            ax.set_xlim(times[0], times[-1])

            for i, line in enumerate(lines):
                line.set_data(times, vels[:, i])
            return lines

        ani = FuncAnimation(fig, update_plot, interval=50)
        plt.show()  # This blocks until you close the window

        rospy.spin()

    except rospy.ROSInterruptException:
        pass