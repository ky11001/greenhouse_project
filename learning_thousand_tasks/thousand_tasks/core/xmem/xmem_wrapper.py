"""
XMem Wrapper for Video Object Segmentation

This wrapper provides a simple interface to XMem for tracking objects through video.
XMem must be installed separately from: https://github.com/hkchengrex/XMem

Installation:
    1. Clone XMem repo: git clone https://github.com/hkchengrex/XMem.git
    2. Download model weights to XMem/saves/XMem.pth
    3. Set XMEM_PATH environment variable or place XMem in expected location

For BC training demonstrations, we need per-timestep segmentation masks.
This wrapper uses XMem to track objects from an initial segmentation.
"""

import sys
import os
from os.path import join
import numpy as np
import torch
from PIL import Image

# Try to find XMem installation
XMEM_PATH = os.environ.get('XMEM_PATH')
if XMEM_PATH is None:
    # Try common locations
    from thousand_tasks.core.globals import ASSETS_DIR
    possible_paths = [
        join(ASSETS_DIR, '..', '..', 'XMem'),  # learning_thousand_tasks/../XMem
        join(ASSETS_DIR, '..', 'XMem'),  # learning_thousand_tasks/XMem
        '/workspace/XMem',  # Docker default
    ]
    for path in possible_paths:
        if os.path.exists(path):
            XMEM_PATH = path
            break

if XMEM_PATH is None:
    raise ImportError(
        "XMem not found. Please install from https://github.com/hkchengrex/XMem\n"
        "Either:\n"
        "  1. Set XMEM_PATH environment variable\n"
        "  2. Clone XMem to project root or assets directory"
    )

sys.path.insert(0, XMEM_PATH)

try:
    from model.network import XMem
    from inference.inference_core import InferenceCore
    from inference.interact.interactive_utils import (
        image_to_torch,
        index_numpy_to_one_hot_torch,
        torch_prob_to_numpy_mask,
        overlay_davis
    )
except ImportError as e:
    raise ImportError(
        f"Failed to import XMem modules from {XMEM_PATH}.\n"
        f"Make sure XMem is properly installed: https://github.com/hkchengrex/XMem\n"
        f"Error: {e}"
    )


class XMemTracker:
    """
    Wrapper for XMem video object segmentation.

    Usage:
        # Initialize tracker
        xmem = XMemTracker(device='cuda')

        # Provide first frame with segmentation mask
        xmem.initialise(rgb=first_frame, segmap=initial_mask)

        # Track through subsequent frames
        for frame in video_frames:
            mask = xmem.compute_object_segmap(frame)
    """

    def __init__(self, single_object=False, device='cuda'):
        """
        Initialize XMem tracker.

        Args:
            single_object: Whether tracking single object (True) or multiple (False)
            device: 'cuda' or 'cpu'
        """
        torch.set_grad_enabled(False)

        self.device = torch.device(device)

        xmem_config = {
            "single_object": single_object,
            "top_k": 30,
            "mem_every": 5,
            "deep_update_every": -1,
            "enable_long_term": True,
            "enable_long_term_count_usage": True,
            "num_prototypes": 128,
            "min_mid_term_frames": 5,
            "max_mid_term_frames": 10,
            "max_long_term_elements": 10000,
        }

        model_path = join(XMEM_PATH, "saves", "XMem.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"XMem model weights not found at {model_path}\n"
                f"Download from: https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem.pth\n"
                f"Place in: {join(XMEM_PATH, 'saves', 'XMem.pth')}"
            )

        self.model = XMem(xmem_config, model_path).eval().to(self.device)
        self.processor = InferenceCore(self.model, xmem_config)

        # Initialization state
        self._num_objects = None
        self._init_segmap = None
        self._init_segmap_torch = None
        self.is_initialised = False

    def initialise(self, rgb, segmap):
        """
        Initialize tracker with first frame and segmentation mask.

        Args:
            rgb: RGB image as numpy array (H, W, 3)
            segmap: Segmentation mask as numpy array (H, W) with integer labels
                    - 0: background
                    - 1, 2, 3, ...: object IDs
        """
        self._num_objects = len(np.unique(segmap)) - 1
        self._init_segmap = segmap.copy()
        self._init_segmap_torch = index_numpy_to_one_hot_torch(
            segmap.astype(np.uint8), self._num_objects + 1
        ).to(self.device)
        self.processor.set_all_labels(range(1, self._num_objects + 1))

        with torch.cuda.amp.autocast(enabled=True):
            frame_torch, _ = image_to_torch(rgb, self.device)
            _ = self.processor.step(frame_torch, self._init_segmap_torch[1:])

        self.is_initialised = True

    def compute_object_segmap(self, rgb):
        """
        Track objects in a new frame.

        Args:
            rgb: RGB image as numpy array (H, W, 3)

        Returns:
            segmap: Boolean segmentation mask (H, W) - True for object, False for background
        """
        if not self.is_initialised:
            raise RuntimeError(
                'XMem is not initialized. Call initialise() with first frame before tracking.'
            )

        torch.cuda.empty_cache()

        with torch.cuda.amp.autocast(enabled=True):
            frame_torch, _ = image_to_torch(rgb, self.device)
            prediction = self.processor.step(frame_torch)
            prediction = torch_prob_to_numpy_mask(prediction)

        # Convert to boolean: object pixels (>0) = True, background (0) = False
        return prediction.astype(bool)
