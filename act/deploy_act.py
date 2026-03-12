#!/usr/bin/env python3
import torch
import numpy as np
import rospy
import cv2
import pickle
from sensor_msgs.msg import JointState, Image
from cv_bridge import CvBridge
from collections import deque

# Import your model definition
from policy import ACTPolicy

# --- CONFIGURATION ---
checkpoint_path = 'my_checkpoints/best_ckpt.pth'
stats_path = 'my_checkpoints/dataset_stats.pkl'

policy_config = {
    'lr': 1e-5,
    'num_queries': 50,    # chunk_size
    'kl_weight': 10,
    'state_dim': 7,       # Sawyer 7 joints
    'hidden_dim': 512,
    'dim_feedforward': 3200,
    'lr_backbone': 1e-5,
    'backbone': 'resnet18',
    'enc_layers': 4,
    'dec_layers': 7,
    'nheads': 8,
    'camera_names': ['head'], 
}

class SawyerACTDeploy:
    def __init__(self):
        rospy.init_node('act_deployment_sawyer')
        
        # 1. Load Model
        self.policy = ACTPolicy(policy_config)
        self.policy.load_state_dict(torch.load(checkpoint_path))
        self.policy.cuda().eval()
        
        # 2. Load Stats
        with open(stats_path, 'rb') as f:
            self.stats = pickle.load(f)
        
        self.bridge = CvBridge()
        self.curr_qpos = None
        self.curr_image = None
        
        # Temporal Aggregation Buffers
        self.chunk_size = policy_config['num_queries']
        self.state_dim = policy_config['state_dim']
        self.all_time_actions = deque(maxlen=self.chunk_size)
        
        # ROS Setup
        rospy.Subscriber('/robot/joint_states', JointState, self.qpos_callback)
        rospy.Subscriber('/io/internal_camera/head_camera/image_raw', Image, self.image_callback)
        self.joint_pub = rospy.Publisher('/robot/limb/right/joint_command', JointState, queue_size=10) # Adjust topic per SDK
        
        self.rate = rospy.Rate(20) # Match your training DT (e.g., 20Hz)
        print("Model loaded and ROS subscribers initialized. Ready.")

    def qpos_callback(self, msg):
        # Ensure joint order matches your training JOINT_NAMES
        # Usually: [j0, j1, j2, j3, j4, j5, j6]
        self.curr_qpos = np.array(msg.position[:7])

    def image_callback(self, msg):
        # Convert ROS image to OpenCV, then to Tensor
        cv_img = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        cv_img = cv2.resize(cv_img, (640, 480)) # Match training resolution
        img_tensor = torch.from_numpy(cv_img).permute(2, 0, 1).float() / 255.0
        self.curr_image = img_tensor.unsqueeze(0).unsqueeze(0).cuda() # (1, 1, C, H, W)

    def get_aggregated_action(self, new_action_chunk):
        """
        Applies Temporal Aggregation:
        $$a_t = \frac{\sum_{i=0}^{k-1} w_i \cdot \hat{a}_{t-i, t}}{\sum w_i}$$
        where $w_i = e^{-m \cdot i}$
        """
        self.all_time_actions.append(new_action_chunk)
        
        num_chunks = len(self.all_time_actions)
        actions_at_curr_step = []
        
        for i in range(num_chunks):
            # The i-th chunk was generated (num_chunks - 1 - i) steps ago
            step_idx = num_chunks - 1 - i
            if step_idx < self.chunk_size:
                actions_at_curr_step.append(self.all_time_actions[i][step_idx])
        
        actions_at_curr_step = torch.stack(actions_at_curr_step)
        
        # Exponential weights (m=0.01)
        k = 0.01
        weights = torch.exp(-k * torch.arange(len(actions_at_curr_step))).cuda()
        weights = weights / weights.sum()
        
        weighted_action = (actions_at_curr_step * weights.unsqueeze(1)).sum(dim=0)
        return weighted_action

    def run(self):
        while not rospy.is_shutdown():
            if self.curr_qpos is None or self.curr_image is None:
                continue

            # 1. Pre-process qpos
            qpos_norm = (self.curr_qpos - self.stats['qpos_mean']) / self.stats['qpos_std']
            qpos_tensor = torch.from_numpy(qpos_norm).float().cuda().unsqueeze(0)

            # 2. Inference
            with torch.inference_mode():
                # Model returns (Batch, Chunk, Dim)
                action_chunk = self.policy(qpos_tensor, self.curr_image) 
            
            # 3. Temporal Aggregation
            raw_action = self.get_aggregated_action(action_chunk[0])
            
            # 4. Post-process to Joint Angles
            action_np = raw_action.cpu().numpy()
            target_qpos = (action_np * self.stats['action_std']) + self.stats['action_mean']

            # 5. Publish Command
            cmd = JointState()
            cmd.position = target_qpos.tolist()
            self.joint_pub.publish(cmd)

            self.rate.sleep()

if __name__ == '__main__':
    try:
        node = SawyerACTDeploy()
        node.run()
    except rospy.ROSInterruptException:
        pass