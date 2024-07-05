# Quantum Machine Learning Classification of Liquid Crystals👋

## Overview

This project aims to develop a quantum machine learning model to classify different phases of liquid crystals. We utilize an Autoencoder U-Net model to preprocess the images and extract meaningful features. The ground truth of the images is accessed to validate the classification performance..

## Project Structure
data/: Contains the dataset of liquid crystal images.
models/: Contains the implementation of the Autoencoder U-Net and quantum machine learning models.
notebooks/: Jupyter notebooks for data preprocessing, model training, and evaluation.
scripts/: Python scripts for various tasks such as data augmentation and model inference.
README.md: This file, providing an overview of the project and instructions to get started.

##Dataset
The dataset consists of grayscale images of liquid crystals captured under different conditions. The images are preprocessed and normalized before being fed into the model.

##Model
##Autoencoder U-Net
The Autoencoder U-Net model is used to denoise and preprocess the images, helping in extracting relevant features. The architecture is as follows:
**Encoder**: Series of convolutional layers to capture features.
**Bottleneck**: Compressed representation of the input.
**Decoder**: Series of deconvolutional layers to reconstruct the image.

## Quantum Classifier
The quantum classifier leverages quantum circuits to classify the extracted features into different phases of liquid crystals. The classifier is trained and validated using the preprocessed images from the Autoencoder U-Net.

##Setup
##Prerequisites
**Python 3.7+**
**TensorFlow**
**PennyLane**
**OpenCV**
**scikit-learn**
**Matplotlib**

##Installation
1.Clone the repository:
git clone https://github.com/Otieno12/QML-liquid-crystals.git
cd QML-liquid-crystals

import cv2
import os

# Directory containing the images
images_dir = 'data/liquid_crystals/5cb'
images_dir1 = 'data/liquid_crystals/mbba'

# Function to preprocess images
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image at path '{image_path}' could not be loaded.")
    resized_image = cv2.resize(image, (64, 64))  # Resize to 64x64
    normalized_image = resized_image / 255.0
    return normalized_image

# Preprocess all images in the directories
for image_name in os.listdir(images_dir):
    image_path = os.path.join(images_dir, image_name)
    image = preprocess_image(image_path)
    # Save or use the preprocessed image

for image_name in os.listdir(images_dir1):
    image_path = os.path.join(images_dir1, image_name)
    image = preprocess_image(image_path)
    # Save or use the preprocessed image



# Training script
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from model import QCNN  # Ensure you have defined your QCNN model in model.py
from utils import train, evaluate  # Ensure you have defined your training and evaluation functions

# Load your dataset
class LiquidCrystalDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images[idx])
        image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        if self.transform:
            image = self.transform(image)
        label = 0 if '5cb' in self.image_dir else 1  # Simple binary classification
        return image, label

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = LiquidCrystalDataset('data/liquid_crystals/5cb', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define model, loss function, and optimizer
model = QCNN().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
train(model, train_loader, criterion, optimizer, epochs=20)


# Evaluation script
val_dataset = LiquidCrystalDataset('data/liquid_crystals/mbba', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

accuracy, loss = evaluate(model, val_loader, criterion)
print(f'Validation Accuracy: {accuracy:.4f}, Validation Loss: {loss:.4f}')


- **Email:** otienoedwardotieno82@gmail.com
- **LinkedIn:** Edward Otieno
- **Twitter:** Edward Otieno

Feel free to reach out if you have any questions or if you'd like to collaborate on a project!
