from PIL import Image
from models.monoD import depth_pro

# Load model and preprocessing transform
model, transform = depth_pro.create_model_and_transforms()
model.eval()

# Load and preprocess an image.
image_path = "assets/dance/00000.jpg"
image, _, f_px = depth_pro.load_rgb(image_path)
image = transform(image)

# Run inference.
import time
t0 = time.time()
prediction = model.infer(image, f_px=f_px)
depth = prediction["depth"]  # Depth in [m].
focallength_px = prediction["focallength_px"]  # Focal length in pixels.
import cv2
import numpy as np
depth = depth.clamp(0,30).squeeze().detach().cpu().numpy()
depth = (depth - depth.min())/(depth.max()-depth.min()) * 255.0
depth = depth.astype(np.uint8)
depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
cv2.imwrite("depth.png", depth)
print(f"Time: {time.time() - t0:.2f}s")
import pdb; pdb.set_trace()