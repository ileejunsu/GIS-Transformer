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
from keras.applications import imagenet_utils
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
# Configure the hyperparameters of the model.
# Values are from table 4.
patch_size = 4  # 2x2, for the Transformer blocks.
image_size = 256
expansion_factor = 2  # expansion factor for the MobileNetV2 blocks.
#Define the MobileViT architecture
def conv_block(x, filters=16, kernel_size=3, strides=2):
    conv_layer = layers.Conv2D(
        filters, kernel_size, strides=strides, activation=tf.nn.swish, padding="same"
    )
    return conv_layer(x)
# Reference: https://git.io/JKgtC
def inverted_residual_block(x, expanded_channels, output_channels, strides=1):
    m = layers.Conv2D(expanded_channels, 1, padding="same", use_bias=False)(x)
    m = layers.BatchNormalization()(m)
    m = tf.nn.swish(m)
    if strides == 2:
        m = layers.ZeroPadding2D(padding=imagenet_utils.correct_pad(m, 3))(m)
    m = layers.DepthwiseConv2D(
        3, strides=strides, padding="same" if strides == 1 else "valid", use_bias=False
    )(m)
    m = layers.BatchNormalization()(m)
    m = tf.nn.swish(m)
    m = layers.Conv2D(output_channels, 1, padding="same", use_bias=False)(m)
    m = layers.BatchNormalization()(m)
    if tf.math.equal(x.shape[-1], output_channels) and strides == 1:
        return layers.Add()([m, x])
    return m
# Reference:
# https://keras.io/examples/vision/image_classification_with_vision_transformer/
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.swish)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x
def transformer_block(x, transformer_layers, projection_dim, num_heads=2):
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, x])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=[x.shape[-1] * 2, x.shape[-1]], dropout_rate=0.1,)
        # Skip connection 2.
        x = layers.Add()([x3, x2])
    return x
def mobilevit_block(x, num_blocks, projection_dim, strides=1):
    # Local projection with convolutions.
    local_features = conv_block(x, filters=projection_dim, strides=strides)
    local_features = layers.AveragePooling2D(pool_size=(2, 2), strides=strides, padding='same')(local_features)
    # Unfold into patches and then pass through Transformers.
    num_patches = int((local_features.shape[1] * local_features.shape[2]) / patch_size)
    non_overlapping_patches = layers.Reshape((patch_size, num_patches, projection_dim))(
        local_features
    )
    global_features = transformer_block(
        non_overlapping_patches, num_blocks, projection_dim
    )
    # Fold into conv-like feature-maps.
    folded_feature_map = layers.Reshape((*local_features.shape[1:-1], projection_dim))(
        global_features
    )
    # Apply point-wise conv -> concatenate with the input features.
    folded_feature_map = conv_block(
        folded_feature_map, filters=x.shape[-1], kernel_size=1, strides=strides
    )
    x = layers.UpSampling2D(size=(1,1),interpolation="bilinear")(x)
    local_global_features = layers.Concatenate(axis=-1)([x, folded_feature_map])
    # Fuse the local and global features using a convoluion layer.
    local_global_features = conv_block(local_global_features, filters=projection_dim, strides=strides)
    return local_global_features
#Create the MobileViT model
def create_mobilevit(num_classes=30):
    inputs = keras.Input((image_size, image_size, 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    # Initial conv-stem -> MV2 block.
    x = conv_block(x, filters=16)
    x = inverted_residual_block(
        x, expanded_channels=16 * expansion_factor, output_channels=16
    )
    # Downsampling with MV2 block.
    x = inverted_residual_block(
        x, expanded_channels=16 * expansion_factor, output_channels=24, strides=2
    )
    x = inverted_residual_block(
        x, expanded_channels=24 * expansion_factor, output_channels=24
    )
    x = inverted_residual_block(
        x, expanded_channels=24 * expansion_factor, output_channels=24
    )
    # First MV2 -> MobileViT block.
    x = inverted_residual_block(
        x, expanded_channels=24 * expansion_factor, output_channels=48, strides=2
    )
    x = mobilevit_block(x, num_blocks=2, projection_dim=64)
    # Second MV2 -> MobileViT block.
    x = inverted_residual_block(
        x, expanded_channels=64 * expansion_factor, output_channels=64, strides=2
    )
    x = mobilevit_block(x, num_blocks=4, projection_dim=80)
    # Third MV2 -> MobileViT block.
    x = inverted_residual_block(
        x, expanded_channels=80 * expansion_factor, output_channels=80, strides=2
    )
    x = mobilevit_block(x, num_blocks=3, projection_dim=96)
    x = conv_block(x, filters=320, kernel_size=1, strides=1)
    # Classification head.
    x = layers.GlobalAvgPool2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)
mobilevit_xxs = create_mobilevit()
mobilevit_xxs.summary()
#Preparing the dataset
batch_size = 64
auto = tf.data.AUTOTUNE
resize_bigger = 280
num_classes = 30
def preprocess_dataset(is_training=True):
    def _pp(image, label):
        if is_training:
            # Resize to a bigger spatial resolution and take the random
            # crops.
            image = tf.image.resize(image, (resize_bigger, resize_bigger))
            image = tf.image.random_crop(image, (image_size, image_size, 3))
            image = tf.image.random_flip_left_right(image)
        else:
            image = tf.image.resize(image, (image_size, image_size))
        label = tf.one_hot(label, depth=num_classes)
        return image, label
    return _pp
def prepare_dataset(dataset, is_training=True):
    if is_training:
        dataset = dataset.shuffle(batch_size * 10)
    dataset = dataset.map(preprocess_dataset(is_training), num_parallel_calls=auto)
    return dataset.batch(batch_size).prefetch(auto)
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
val_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
train_dataset = prepare_dataset(train_dataset, is_training=True)
val_dataset = prepare_dataset(val_dataset, is_training=False)
#Training a MobileViT model
learning_rate = 0.001
label_smoothing_factor = 0.1
epochs = 500
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing_factor)
#Save the initial weights inside the initial_weights file
initial_weights = os.path.join("/data/groups/gomesr/REU_2023/RaviGadgil", 'initial_weights')
def run_experiment(epochs=epochs):
    mobilevit_xxs = create_mobilevit(num_classes=num_classes)
    #Saving training data(parameters and performance) to CSV file
    parameters_log = CSVLogger('MobileViTReducedV1_AID_6040_Parameter_log.csv')
    mobilevit_xxs.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])
    mobilevit_xxs.save_weights(initial_weights)
    checkpoint_filepath = "/tmp/checkpoint"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )
    mobilevit_xxs.load_weights(initial_weights)
    #Record the time for the model training
    start = time()
    mobilevit_xxs.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[checkpoint_callback],
    )
    print("\n"+str(time()-start))
    _, accuracy = mobilevit_xxs.evaluate(val_dataset)
    print(f"Validation accuracy: {round(accuracy * 100, 2)}%")
    return mobilevit_xxs
model = run_experiment()
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
save_test.tofile("MobileViTReducedV1_AID_6040_test_labels.csv", sep=",")
save_predicted = np.array(predicted_labels)
save_predicted.tofile("MobileViTReducedV1_AID_6040_predicted_labels.csv", sep=",")
# Display the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()
plt.savefig("Confusion_Matrix.png", dpi=300)
#Save the model
tf.keras.models.save_model(model, 'MobileViTReducedV1-AID60_40.h5')