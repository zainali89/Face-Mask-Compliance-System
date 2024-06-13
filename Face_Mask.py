#!/usr/bin/env python
# coding: utf-8

# ## FACE MASK DETECTION

# Importing Libraries
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import cv2
import seaborn as sns
from PIL import Image
import dask.bag as bag
from dask import diagnostics
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc

# Setting the path of training and testing directories
TRAIN_PATH = "C:/Users/Face_Mask/Train/"
TEST_PATH = "C:/Users/Face_Mask/Test/"
VAL_PATH = "C:/Users/Face_Mask/Validation/"

# Display folder names in the training directory
print("Folders in training directory:")
for folder in os.listdir(TRAIN_PATH):
    print(folder)

# Get the list of categories in the training path
categories = os.listdir(TRAIN_PATH)
print('Categories:', categories)

# Create labels for categories
labels = [0, 1]
label_dict = dict(zip(categories, labels))
print('Label Dictionary:', label_dict)
print('Number of Classes:', len(label_dict))

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

# For Mask Type
img = keras.preprocessing.image.load_img(TRAIN_PATH + 'Mask/0003.jpg')
x = keras.preprocessing.image.img_to_array(img)
x = x.reshape((1,) + x.shape)

# Augment a single image
i = 0
for batch in datagen.flow(x, batch_size=1, save_to_dir='C:/Users/Face_Mask/Mask Augmented', save_format='jpeg'):
    i += 1
    if i == 15:
        break

# Display the original image
img = mpimg.imread(TRAIN_PATH + 'Mask/0003.jpg')
plt.imshow(img)
plt.show()

# Augment all images in 'Mask' directory of Training Set
for filename in os.listdir(TRAIN_PATH + 'Mask/'):
    img = keras.preprocessing.image.load_img(TRAIN_PATH + 'Mask/' + filename)
    x = keras.preprocessing.image.img_to_array(img)
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=TRAIN_PATH + 'Mask', save_format='jpeg'):
        i += 1
        if i == 15:
            break
print('Mask augmentation done')

# For Non Mask Type
img = keras.preprocessing.image.load_img(TRAIN_PATH + 'Non Mask/0.jpg')
x = keras.preprocessing.image.img_to_array(img)
x = x.reshape((1,) + x.shape)

# Augment a single image
i = 0
for batch in datagen.flow(x, batch_size=1, save_to_dir='C:/Users/Face_Mask/Non Mask Augmented', save_format='jpeg'):
    i += 1
    if i == 15:
        break

# Display the original image
img = mpimg.imread(TRAIN_PATH + 'Non Mask/0.jpg')
plt.imshow(img)
plt.show()

# Augment all images in 'Non Mask' directory of Training Set
for filename in os.listdir(TRAIN_PATH + 'Non Mask/'):
    img = keras.preprocessing.image.load_img(TRAIN_PATH + 'Non Mask/' + filename)
    x = keras.preprocessing.image.img_to_array(img)
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=TRAIN_PATH + 'Non Mask', save_format='jpeg'):
        i += 1
        if i == 10:
            break
print('Non Mask augmentation done')

# Plotting number of images by class
number_classes = {folder: len(os.listdir(TRAIN_PATH + folder)) for folder in os.listdir(TRAIN_PATH)}
df = pd.DataFrame({'Type': number_classes.keys(), 'Total': number_classes.values()})

sns.barplot(x='Type', y='Total', data=df)
plt.title("Number of Images by Class")
plt.xlabel('Class Name')
plt.ylabel('# Images')
plt.show()

# Define directories for Mask and Non Mask images
directories = {'Mask': TRAIN_PATH + 'Mask/', 'Non Mask': TRAIN_PATH + 'Non Mask/'}

# Function to get image dimensions
def get_dims(file):
    im = Image.open(file)
    arr = np.array(im)
    h, w, d = arr.shape
    return h, w

# Plotting image sizes for each class
for n, d in directories.items():
    filepath = d
    filelist = [filepath + f for f in os.listdir(filepath)]
    dims = bag.from_sequence(filelist).map(get_dims)
    with diagnostics.ProgressBar():
        dims = dims.compute()
    dim_df = pd.DataFrame(dims, columns=['height', 'width'])
    sizes = dim_df.groupby(['height', 'width']).size().reset_index().rename(columns={0: 'count'})
    sizes.plot.scatter(x='width', y='height')
    plt.title(f'Image Sizes (pixels) | {n}')
    plt.show()

# Function to load the images and their labels
def load_data(path, label):
    X = []  # list of images
    y = []  # list of labels
    for filename in os.listdir(path):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(path, filename)
            image = Image.open(image_path)
            image = image.resize((224, 224))  # Resize the image to a standard size
            image = image.convert("L")  # Convert the image to grayscale
            image = np.array(image) / 255.0  # Normalize the pixel values
            X.append(image)
            y.append(label)
    return X, y

# Load test data for both Mask and Non Mask categories
mask_path = os.path.join(TEST_PATH, "Mask")
non_mask_path = os.path.join(TEST_PATH, "Non Mask")

X_test_mask, y_test_mask = load_data(mask_path, 1)
X_test_non_mask, y_test_non_mask = load_data(non_mask_path, 0)

# Combine the images and labels into a single dataset
X_test = X_test_mask + X_test_non_mask
y_test = y_test_mask + y_test_non_mask

# Define the data augmentation pipeline for training data
BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_CLASSES = 2
NUM_EPOCHS = 10

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# Load the training data using the data augmentation pipeline
train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

# Define the data augmentation pipeline for validation data
val_datagen = ImageDataGenerator(rescale=1./255)

# Load the validation data using the data augmentation pipeline
val_generator = val_datagen.flow_from_directory(
    VAL_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

# Training the Model
# Load the pre-trained InceptionV3 model
base_model = tf.keras.applications.InceptionV3(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,
    weights='imagenet')

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
predictions = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)

# Create the final model
model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

# Compile the model with categorical cross-entropy loss and Adam optimizer
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

# Train the model on the training data
history = model.fit(
    train_generator,
    epochs=NUM_EPOCHS,
    validation_data=val_generator)

# Evaluate the model on the validation data
val_loss, val_acc = model.evaluate(val_generator)
print('Validation accuracy:', val_acc)

# Training & Validation Accuracy and Loss Graph
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, label='Training Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# Predicting the Testing Data
IMAGE_SIZE = (224, 224)

# Create a test data generator with no data augmentation
test_datagen = ImageDataGenerator(rescale=1./255)

# Load the test data using the generator
test_generator = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False)

# Make predictions on the test data
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_test, predicted_classes)
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, linewidths=0.5, linecolor="red", fmt=".0f")
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel("True Label")
plt.show()

# Classification Report
report = classification_report(y_test, predicted_classes)
print(report)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, predicted_classes)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
