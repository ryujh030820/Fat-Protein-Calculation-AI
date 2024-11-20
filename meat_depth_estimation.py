import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T

# Load a pretrained MiDaS model for depth estimation
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
midas.eval()

# Load transforms to convert the image into the format expected by the MiDaS model
transform = torch.hub.load("intel-isl/MiDaS", "transforms").default_transform

# Load the image
image_path = "background_removal/data/15.jpg"  # 이미지 경로를 본인의 경로로 변경
image = Image.open(image_path).convert("RGB")  # Ensure image is in RGB format

# Resize the image so that its dimensions are divisible by 32
resize_transform = T.Resize((int(np.ceil(image.size[1] / 32) * 32),
                             int(np.ceil(image.size[0] / 32) * 32)))
image_resized = resize_transform(image)

# Convert image to NumPy array and normalize to [0, 1]
image_np = np.asarray(image_resized) / 255.0  # Convert to NumPy array and normalize

# Convert NumPy array to PyTorch Tensor and permute dimensions to [C, H, W]
image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()

# Add a batch dimension [B, C, H, W]
input_batch = image_tensor.unsqueeze(0)

# Perform depth estimation
with torch.no_grad():
    prediction = midas(input_batch)

# Resize depth map to original image size and normalize
depth_map = torch.nn.functional.interpolate(
    prediction.unsqueeze(1),
    size=image.size[::-1],
    mode="bicubic",
    align_corners=False,
).squeeze()

# Normalize depth map to [0, 1] for visualization
depth_map = depth_map.numpy()
depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))

# Display the depth map
plt.imshow(depth_map, cmap='inferno')
plt.axis('off')
plt.title("Estimated Depth Map of the Meat")
plt.show()

# Step 1: Hand Length in Pixels
# Assuming we know the real length of the hand (e.g., 18 cm)
real_hand_length_cm = 18.0

# Assume you have identified the hand in the image and measured its length in pixels
hand_length_pixels = 50  # Replace with the actual pixel length from the image

# Calculate the scale ratio (cm per pixel)
scale_ratio = real_hand_length_cm / hand_length_pixels

# Step 2: Select Region of Interest (ROI) for meat thickness estimation
# Assuming we select a central region of the depth map to estimate thickness
height, width = depth_map.shape
roi_top = int(height * 0.4)
roi_bottom = int(height * 0.6)
roi_left = int(width * 0.4)
roi_right = int(width * 0.6)

# Extract ROI from the depth map
roi = depth_map[roi_top:roi_bottom, roi_left:roi_right]

# Estimate average thickness from the ROI in relative units
average_thickness_relative = np.mean(roi)

# Step 3: Convert relative thickness to actual thickness (in cm)
# Note: Depth map values are normalized, so we use them as relative depth
# Assuming the max depth represents a thickness of approximately 10 cm
max_thickness_estimate_cm = 10.0
estimated_thickness_cm = average_thickness_relative * max_thickness_estimate_cm

# Apply scale ratio to convert to real-world thickness
real_thickness_cm = estimated_thickness_cm * scale_ratio

# Display the estimated real thickness
print(f"Estimated Average Thickness (in cm): {real_thickness_cm:.2f}")