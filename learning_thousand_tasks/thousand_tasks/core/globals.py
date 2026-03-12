"""
Global configuration for the learning_thousand_tasks repository.

This file contains paths and parameters used throughout the codebase.
Update these paths to match your setup.
"""

from pathlib import Path
import numpy as np

# ==============================================================================
# Directory Paths
# ==============================================================================

# Get the repository root (learning_thousand_tasks/)
REPO_ROOT = Path(__file__).parent.parent.parent

# Assets directory containing calibration files, model weights, and example data
ASSETS_DIR = REPO_ROOT / 'assets'

# Tasks directory (contains task demonstrations and data)
# Used by hierarchical retrieval system
TASKS_DIR = ASSETS_DIR / 'demonstrations'

# Training data directory (contains demonstration datasets)
# Default: assets/training_data/
# For your own data, change this to point to your demonstrations
TRAINING_DATA_DIR = ASSETS_DIR / 'training_data'

# Dataset directory for retrieval/training datasets
DATASET_DIR = TRAINING_DATA_DIR

# Testing data directory (contains rollout/evaluation data)
# Default: assets/testing_data/
TESTING_DATA_DIR = ASSETS_DIR / 'testing_data'

# Directory for trained model checkpoints
# All trained models are stored in assets/checkpoints/
CHECKPOINTS_DIR = ASSETS_DIR / 'checkpoints'

# Directory for training outputs (logs, wandb runs, etc.)
# All training outputs go to assets/outputs/
OUTPUTS_DIR = ASSETS_DIR / 'outputs'

# Directory for training runs (backward compatibility)
RUNS_DIR = ASSETS_DIR / 'runs'

# ==============================================================================
# Robot Hardware Parameters
# ==============================================================================
# These parameters are from the Sawyer robot used in the original paper.
# If deploying on a different robot, update these accordingly or replace
# with your robot-specific parameters.

# Robot control rate (Hz)
SAWYER_CONTROL_RATE = 30

# Maximum joint speed for safety
MAX_ROBOT_JOINT_SPEED = 0.1

# Pre-defined joint configurations for specific poses
# These are Sawyer-specific joint angles (7-DOF)
NO_OCCLUSIONS_JOINT_ANGLES = np.array(
    [0.62110547, -1.56480371, -0.20711914, 1.42105859, 0.06802441, 1.66493945, 0.21474707])

WRIST_CAM_NEXT_TO_HEAD_JOINT_ANGLES = np.array(
    [0.80218652, -1.83028223, 0.41008203, 1.68355957, -0.56745117, 1.87058496, 1.73311328])

# Head camera pan angle (radians)
HEAD_PAN_ANGLE = -4 * np.pi / 180
