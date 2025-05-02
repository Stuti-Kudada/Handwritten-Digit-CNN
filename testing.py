import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from google.colab import files

# Load the trained model
MODEL_PATH = "mnist_cnn_model.h5"
if os.path.exists(MODEL_PATH):
    model = keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
else:
    print("Model not found! Train the model first.")
    exit()

# Function to preprocess image
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    image = cv2.resize(image, (28, 28))  # Resize to 28x28
    image = 255 - image  # Invert colors (MNIST digits are white on black)
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize
    image = image.reshape(1, 28, 28, 1)  # Reshape for CNN input
    return image, image.squeeze()  # Return both normalized and processed images

# Upload and predict multiple images
print("Please upload one or more images:")
uploaded = files.upload()

for image_path in uploaded.keys():
    image, processed_image = preprocess_image(image_path)  # Preprocess the image
    prediction = model.predict(image)
    predicted_label = int(np.argmax(prediction))

    # Display the processed image
    plt.imshow(processed_image, cmap='gray')
    plt.title(f"Predicted Digit: {predicted_label}")
    plt.show()

    print(f"Image: {image_path}, Predicted Digit: {predicted_label}")