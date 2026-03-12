#!/usr/bin/env python3
import sys

# --- HACK TO SILENCE ARGPARSE ---
# This prevents policy.py from trying to read command line args and crashing
sys.argv = ['act_ros_node.py']

import rospy
import torch
import numpy as np
import pickle
import cv2
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
import torchvision.transforms as transforms

# --- IMPORTS FROM YOUR FOLDER ---
from policy import ACTPolicy 

# --- FIXED CONFIG CLASS ---
# Inherits from dict so .items() works
# Sets __dict__ so .dot notation works
class ModelArgs(dict):
    def __init__(self, *args, **kwargs):
        super(ModelArgs, self).__init__(*args, **kwargs)
        self.__dict__ = self

class ACTNode:
    def __init__(self):
        rospy.init_node('act_inference_node')
        
        # 1. CONFIGURATION
        # This acts as BOTH a dictionary and an object
        self.args = ModelArgs({
            'lr': 1e-5, 
            'num_queries': 50,
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
            'batch_size': 1,      # Added to prevent other potential errors
            'weight_decay': 1e-4, # Added just in case
        })

        # Files
        self.ckpt_path = 'policy_best.ckpt' 
        self.stats_path = 'dataset_stats.pkl' 
        
        # Setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bridge = CvBridge()
        
        # Load the Brain
        self.setup_model()
        
        # --- ROS CONNECTIONS ---
        self.pub = rospy.Publisher('/desired_velocity', Twist, queue_size=1)
        
        # NOTE: Verify this topic matches your RealSense!
        rospy.Subscriber('/camera/color/image_raw', Image, self.img_callback) 
        rospy.Subscriber('/robot/joint_states', JointState, self.joint_callback)
        
        # State Storage
        self.curr_img = None
        self.curr_qpos = None
        self.arm_joints = ['right_j0', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 'right_j5', 'right_j6']
        
        self.normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.rate = rospy.Rate(15) 

    def setup_model(self):
        rospy.loginfo(f"Loading Model from {self.ckpt_path}...")
        
        # 1. Load Statistics
        with open(self.stats_path, 'rb') as f:
            self.stats = pickle.load(f)
            
        # 2. Initialize Model Architecture
        self.policy = ACTPolicy(self.args)
        
        # 3. Load Weights
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
            pass

    def joint_callback(self, msg):
        try:
            val_dict = dict(zip(msg.name, msg.position))
            qpos = np.array([val_dict[j] for j in self.arm_joints])
            qpos_norm = (qpos - self.stats['qpos_mean']) / self.stats['qpos_std']
            self.curr_qpos = torch.from_numpy(qpos_norm).float().unsqueeze(0).to(self.device)
        except KeyError:
            pass 

    def run(self):
        rospy.loginfo("ACT Node Running... Waiting for data.")
        while not rospy.is_shutdown():
            if self.curr_img is not None and self.curr_qpos is not None:
                with torch.inference_mode():
                    # Inference
                    all_actions = self.policy(self.curr_qpos, self.curr_img)
                
                # Action Selection (Time t=0)
                raw_action = all_actions[:, 0] 
                raw_action = raw_action.squeeze().cpu().numpy()
                
                # Denormalize
                action = (raw_action * self.stats['action_std']) + self.stats['action_mean']
                
                # Publish
                t = Twist()
                t.linear.x, t.linear.y, t.linear.z = action[0], action[1], action[2]
                t.angular.x, t.angular.y, t.angular.z = action[3], action[4], action[5]
                self.pub.publish(t)
            
            self.rate.sleep()

if __name__ == '__main__':
    try:
        node = ACTNode()
        node.run()
    except rospy.ROSInterruptException:
        pass