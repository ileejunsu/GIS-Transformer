#!/usr/bin/env python
# coding: utf-8
#import all necessary packages and modules
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, losses, Model
from tensorflow import keras
import numpy as np
import os
import tempfile
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import CSVLogger
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns
import tensorflow_addons as tfa
from time import time
#method to load the dataset
def load_dataset(root_dir, image_size):
    image_array = []
    labels = []
    image_labels = {}
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".tif"):
                image_path = os.path.join(root, file)
                label = os.path.basename(root)
                labels.append(label)
                # Load the image 
                image = cv2.imread(image_path)
                #make sure image is 256x256
                image = cv2.resize(image, image_size)
                #add the image to the array
                image_array.append(image)
    # Convert the image array and labels list to NumPy arrays
    image_array = np.array(image_array)
    labels = np.array(labels)
    return image_array, labels
#method to display an image from the dataset with a label
def display_image_with_label(image, label):
    plt.imshow(image)
    plt.axis('off')
    plt.title(label)
    plt.show()
#load the dataset
dataset_path = "/data/groups/rozaripf/REU_2023/data/AID"
image_size = (256, 256)
image_array, labels = load_dataset(dataset_path, image_size)
label_encoder = LabelEncoder() #instance of encoder
#transform the labels to numerical values
num_labels = label_encoder.fit_transform(labels)
print(num_labels)
print(label_encoder.classes_)
#resize the images in the numpy array
image_array.shape
#split the dataset
def split_dataset(image_array, labels, test_split=0.4):
    # Split the data into train and test sets
    train_images, test_images, train_labels, test_labels = train_test_split(
        image_array, num_labels, test_size=test_split, stratify=labels, random_state=42
    )
    return train_images, train_labels, test_images, test_labels
#save the training images and labels as well as the testing images and labels for later usage in the program
train_images, train_labels, test_images, test_labels = split_dataset(image_array, labels)
#get the shape of the training images and labels as well as the testing images and labels
train_images.shape, train_labels.shape, test_images.shape, test_labels.shape
# Display a random image with its label from the train set
import random
random_index = random.randint(0, len(train_images) - 1)
display_image_with_label(train_images[random_index], train_labels[random_index])
#Various variables used to build the model
num_classes = 30
input_shape = (256, 256, 3)
#Configuring the hyperparameters
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 500
image_size = 72  # We'll resize input images to this size
patch_size = 6  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier
#Utilize data augmentation
data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(
            height_factor=0.2, width_factor=0.2
        ),
    ],
    name="data_augmentation",
)
# Compute the mean and the variance of the training data for normalization.
data_augmentation.layers[0].adapt(train_images)
#Implement the multilayer perceptron
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x
#Implement Patch creation as a layer
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
#Display patches for a simple image
import matplotlib.pyplot as plt
plt.figure(figsize=(4, 4))
image = train_images[np.random.choice(range(train_images.shape[0]))]
plt.imshow(image.astype("uint8"))
plt.axis("off")
resized_image = tf.image.resize(
    tf.convert_to_tensor([image]), size=(image_size, image_size)
)
patches = Patches(patch_size)(resized_image)
print(f"Image size: {image_size} X {image_size}")
print(f"Patch size: {patch_size} X {patch_size}")
print(f"Patches per image: {patches.shape[1]}")
print(f"Elements per patch: {patches.shape[-1]}")
n = int(np.sqrt(patches.shape[1]))
plt.figure(figsize=(4, 4))
for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
    plt.imshow(patch_img.numpy().astype("uint8"))
    plt.axis("off")
#Implement the patch encoding layer
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
#Build the Vision Transformer Image model
def create_vit_classifier():
    inputs = layers.Input(shape=input_shape)
    # Augment data.
    augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])
    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.2)
    # Classify outputs.
    logits = layers.Dense(num_classes)(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model
#Save the initial weights inside the initial_weights file
initial_weights = os.path.join("/data/groups/gomesr/REU_2023/RaviGadgil", 'initial_weights')
#Compile, train, and evaluate the Vision Transformer Image model
def run_experiment(model):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )
    #Saving training data(parameters and performance) to CSV file
    parameters_log = CSVLogger('ViT_AID_6040_Parameter_log.csv')
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )
    #See trainable parameters and other features of the model
    model.summary()
    #Save the initial weights from the initial_weights file
    model.save_weights(initial_weights)
    checkpoint_filepath = "/tmp/checkpoint"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )
    #Load the initial weights of the model, so the models start at the same random value
    model.load_weights(initial_weights)
    #Record the time for the model training
    start = time()
    model.fit(
        train_images,
        train_labels,
        epochs=num_epochs,
        validation_data=(test_images, test_labels),
        batch_size=batch_size,
        callbacks=[checkpoint_callback]
    )
    print("\n"+str(time()-start))
    return model
vit_classifier = create_vit_classifier()
model = run_experiment(vit_classifier)
#Get the model predictions on the array of test_images
predictions = model.predict(test_images)
#Get and print out the predicted labels
predicted_labels = np.argmax(predictions, axis=1)
print(predicted_labels)
#calculate the overall accuracy
correct_predictions = np.sum(predicted_labels == test_labels)
total_samples = len(test_labels)
overall_accuracy = correct_predictions / total_samples
#print overall accuracy
print("Overall Accuracy: {:.2%}".format(overall_accuracy))
#Confusion matrix to show accuracy
confusion_mat = confusion_matrix(test_labels, predicted_labels)
#Save the test labels and predicted labels into csv files for later usage
save_test = np.array(test_labels)
save_test.tofile("ViT_AID_6040_test_labels.csv", sep=",")
save_predicted = np.array(predicted_labels)
save_predicted.tofile("ViT_AID_6040_predicted_labels.csv", sep=",")
# Display the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()
#Save the model
tf.keras.models.save_model(model, 'ViT-AID60_40.h5')