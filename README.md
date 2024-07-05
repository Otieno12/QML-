# Hi there, I'm Edward Otieno 👋

## About Me

I am a student of Chemistry and Materials Engineering with a passion for computational modeling and quantum machine learning. My current focus is on classifying liquid crystals using advanced computational techniques.

## Skills

- **Chemistry & Materials Engineering**
  - Computational Chemistry
  - Materials Characterization
  - Liquid Crystals Research

- **Computational Modeling**
  - Molecular Dynamics (GROMACS)
  - Density Functional Theory (DFT)
  - Monte Carlo Simulations

- **Quantum Machine Learning**
  - Quantum Circuits
  - Quantum Data Encoding
  - Qiskit & PennyLane

- **Programming Languages**
  - Python
  - javascript
  - MATLAB
  - R
  - Rust

## Projects

### [QML for Liquid Crystals Classification](https://github.com/Otieno12/QML-)
Developing a quantum machine learning model to classify liquid crystals phases. The project involves using quantum circuits and classical-quantum hybrid models to achieve high accuracy in phase classification.

### [Simulation of Liquid Crystals Using GROMACS](https://github.com/Otieno12/LC-Simulation)
A comprehensive project simulating the behavior of liquid crystals using GROMACS, focusing on phase transitions and molecular dynamics.

## Contact

- **Email:** [your-email@example.com]
- **LinkedIn:** [Your LinkedIn Profile](https://www.linkedin.com/in/your-profile)
- **Twitter:** [@your_twitter_handle](https://twitter.com/your_twitter_handle)

Feel free to reach out if you have any questions or if you'd like to collaborate on a project!

























git clone https://github.com/your-username/liquid-crystal-qcnn.git
cd liquid-crystal-qcnn
pip install -r requirements.txt


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
