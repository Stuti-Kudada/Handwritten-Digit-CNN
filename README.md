# Handwritten Digit Recognition with CNN

This project builds an OCR system using a **Convolutional Neural Network (CNN)** to recognize handwritten digits, trained on the **MNIST dataset**. The model achieves **97.14% accuracy** and supports testing on user-uploaded digit images.

## Table of Contents
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Model Training](#model-training)
- [Model Testing](#model-testing)
- [Requirements](#requirements)
- [Usage](#usage)

## Project Overview

This project implements an OCR model using Convolutional Neural Networks (CNN) for handwritten digit recognition. The model is trained on the MNIST dataset, achieving an accuracy of 97.14%.

Model training involves preprocessing the MNIST dataset, splitting it into training and validation sets, defining a CNN architecture, training the model, and evaluating its performance.

The testing script allows users to upload custom images, which are processed and predicted by the trained model.

## Technologies Used

- Python  
- TensorFlow / Keras  
- OpenCV  
- NumPy  
- Matplotlib

## Model Training

- Data normalization (0–1 scaling) and reshaping for CNN input.
- Model includes convolution, max pooling, and dense layers.
- Used **SGD optimizer** and **sparse categorical crossentropy** loss.
- **80-20 train-validation split** to enhance generalization.
- Model saved as `mnist_cnn_model.h5` after training.
- Achieved **97.14% accuracy** on validation data.

## Model Testing

- Upload and test handwritten digit images.
- Preprocessing includes grayscale conversion, resizing, and normalization.
- Predicts and displays digit using the trained model.

## Requirements

- Python 3.x  
- TensorFlow  
- OpenCV  
- NumPy  
- Matplotlib

Install dependencies with:

```bash
pip install tensorflow opencv-python numpy matplotlib
