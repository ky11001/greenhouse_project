
import numpy as np
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from thousand_tasks.core.globals import ASSETS_DIR
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)
for _ in range(20):
    pipeline.wait_for_frames()
frames = pipeline.wait_for_frames()
rgb   = np.asanyarray(frames.get_color_frame().get_data())[:,:,::-1].copy()
depth = np.asanyarray(frames.get_depth_frame().get_data())
pipeline.stop()

sam = sam_model_registry['vit_h'](checkpoint=str(ASSETS_DIR / 'sam_vit_h_4b8939.pth'))
mask_gen = SamAutomaticMaskGenerator(sam, points_per_side=16)
masks = mask_gen.generate(rgb)

h, w = rgb.shape[:2]
cx, cy = w//2, h//2

print('Found %d masks' % len(masks))
print('%8s  %10s  %8s  %8s  %8s' % ('area','avg_depth','cx_dist','mask_cx','mask_cy'))
for m in sorted(masks, key=lambda x: x['area'], reverse=True)[:15]:
    area = m['area']
    ys, xs = np.where(m['segmentation'])
    mx, my = xs.mean(), ys.mean()
    dist = ((mx-cx)**2 + (my-cy)**2)**0.5
    d = depth[m['segmentation']]
    avg_d = d[d>0].mean() if (d>0).any() else 0
    print('%8d  %10.0f  %8.1f  %8.1f  %8.1f' % (area, avg_d, dist, mx, my))