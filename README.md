# Intergrating deep leep into Classification of Liquid Crystals👋

## Overview
This project aims to develop a deep learning (DL) framework to classify different phases of liquid crystals. We incorporate a Ordinal Neural Network (ONN) to account for the inherent order in the phases (e.g., nematic < smectic < isotropic). To enhance the input quality, we leverage an Resnet 50 to extract features from the images.  The ground truth of the images is used to validate the classification performance, emphasizing the model's ability to respect the ordinal relationships between phases while maintaining high accuracy.

## Project Structure
data/: Contains the dataset of liquid crystal images.
models/: Contains the implementation of DL  models.
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

- **Email:** otienoedwardotieno82@gmail.com
- **LinkedIn:** Edward Otieno
- **Twitter:** Edward Otieno

Feel free to reach out if you have any questions or if you'd like to collaborate on a project!
