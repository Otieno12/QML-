# installing the libraries 
! pip install grad-cam



import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# === Load and prepare model ===
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(class_names))  
model = model.to(device)
model.eval()

# === Load image ===
img_path = "//content/5cb_temp_30.1818.jpeg"
image = Image.open(img_path).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

input_tensor = transform(image).unsqueeze(0).to(device)

# Normalized image for overlay
rgb_img = np.array(image.resize((224, 224))).astype(np.float32) / 255.0

# Set target layer
target_layers = [model.layer4[-1]]  # [-1] for isotropic phase /[-2] for nematic and [-3] for isotropic phase 

# Predict class
with torch.no_grad():
    outputs = model(input_tensor)
    pred_class = outputs.argmax().item()

# Set Grad-CAM target
targets = [ClassifierOutputTarget(pred_class)]

# Apply Grad-CAM
with GradCAM(model=model, target_layers=target_layers) as cam:
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

# Display
plt.imshow(visualization)
plt.axis("off")
plt.title(f"Grad-CAM (Predicted: {class_names[pred_class]})")
plt.show()
