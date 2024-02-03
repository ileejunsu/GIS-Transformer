import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import os
from tensorflow.image import resize

def load_and_prepare_image(img_path, target_size=(256, 256)):
    """Load and prepare an image for prediction, resizing if necessary."""
    # Load the image
    img = load_img(img_path)
    img = img_to_array(img)

    # Resize the image to the target size
    img_resized = resize(img, target_size)

    # Scale the image pixels to [0, 1]
    img_resized /= 255.0

    # Expand the dimensions to match the input shape (None, height, width, channels)
    img_ready = np.expand_dims(img_resized, axis=0)
    return img_ready

def predict_image(model_path, img_folder):
    """Load a model and predict the class of images in a folder."""
    # Load the model
    model = load_model(model_path)

    # Predict and print results for each image in the folder
    for img_name in os.listdir(img_folder):
        img_path = os.path.join(img_folder, img_name)
        if os.path.isfile(img_path):
            img_array = load_and_prepare_image(img_path)
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)  # Assuming model outputs class probabilities
            print(f"Image: {img_name}, Predicted Class: {predicted_class}")

# Example usage
model_name = 'MobileViT-AID60_40.h5'  # Change this to your model file name
#folder_name = '/path/to/image/folder'  # Change this to your image folder path
folder_name = 'Demo_Images\AID'  # Change this to your image folder path

model_path = os.path.join('h5', model_name)  # Assuming the model is in a folder named 'h5'

predict_image(model_path, folder_name)