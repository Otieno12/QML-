import os
import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, fftshift


def apply_fourier_transform(image):
    """Computes 2D Fourier Transform of an image."""
    f_transform = fft2(image)
    f_shift = fftshift(f_transform)
    magnitude_spectrum = np.log1p(np.abs(f_shift))
    return magnitude_spectrum



class LiquidCrystalDataset(Dataset):
    def __init__(self, json_file, img_dir, transform=None):
        """
        Args:
            json_file (str): Path to the JSON file with labels.
            img_dir (str): Directory containing images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        with open(json_file, "r") as f:
            self.data = json.load(f)
        self.img_dir = img_dir
        self.transform = transform
        self.label_map = {"Nematic": 0, "Isotropic": 1}  # Encoding labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_key = list(self.data.keys())[idx]
        sample = self.data[sample_key]
        
        # Load Original Image
        img_path = os.path.join(self.img_dir, sample["original image"])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply Fourier Transform
        fourier_img = apply_fourier_transform(image)

        # Normalize images
        image = image / 255.0
        fourier_img = fourier_img / np.max(fourier_img)

        # Convert to tensors
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        fourier_img = torch.tensor(fourier_img, dtype=torch.float32).unsqueeze(0)

        # Get Label
        label = self.label_map[sample["phase"]]
        label = torch.tensor(label, dtype=torch.long)

        return {"image": image, "fourier_img": fourier_img, "label": label}




# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts image to tensor
])

# Load dataset
dataset = LiquidCrystalDataset(json_file="dataset.json", img_dir="path_to_images", transform=transform)

# Split dataset into training (70%) and validation (30%)
train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# Get a batch from DataLoader
data_iter = iter(train_loader)
batch = next(data_iter)

# Extract sample image and its Fourier transform
sample_img = batch["image"][0].squeeze().numpy()
sample_fourier = batch["fourier_img"][0].squeeze().numpy()

# Display images
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(sample_img, cmap="gray")
axs[0].set_title("Original Image")

axs[1].imshow(sample_fourier, cmap="gray")
axs[1].set_title("Fourier Transform")

plt.show()
