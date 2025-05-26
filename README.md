# Integrating deep leep into Classification of Liquid Crystals👋

## Overview
This project aims to develop a deep learning (DL) framework to classify different phases of liquid crystals. We incorporate a Deep Neural Network (DNN) -to classify (e.g., nematic < smectic < isotropic). To enhance the input quality, we leverage an Resnet 50 to extract features from the images.  The ground truth of the images is used to validate the classification performance, Grad-Cam Approach is employed to add the explainability of the model as well as achieving high accuracy for the predictions.

## Project Structure
data/: Contains the dataset of liquid crystal images.
models/: Contains the implementation of DL  models.
notebooks/: Jupyter notebooks for data preprocessing, model training, and evaluation.
scripts/: Python scripts for various tasks such as data augmentation and model inference.
README.md: This file, providing an overview of the project and instructions to get started.

##Dataset
The dataset consists images of liquid crystals captured under different conditions. The images are preprocessed and normalized before being fed into the model.

##Model
ResNet 50  



##Setup
##Prerequisites
**Python 3.7+**
**TensorFlow**
**Torch**
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
