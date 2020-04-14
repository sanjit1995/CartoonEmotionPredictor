import tensorflow as tf
from tensorflow import keras
import kerastuner
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from CartoonEmotionsPredictor.imageOperations import convertVideoToImages, augmentData
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import shutil
import os
import cv2 as cv
from sklearn.preprocessing import LabelEncoder

# print(tf.keras.version)

# define framerate
framerate = 5

# Import datasets
train_dataset = pd.read_csv("data/Train.csv")
test_dataset = pd.read_csv("data/Test.csv")

# Import labels and names
train_filenames = train_dataset.iloc[:, 0]
temp_train_labels = train_dataset.iloc[:, 1]

# Encode the labels to numbers
label_encoder = LabelEncoder()
categorical_train_labels = label_encoder.fit_transform(temp_train_labels)

# print(categorical_train_labels[0:50])
# exit(0)
# print(train_labels)

# Extract filenames from test-dataset
test_filenames = test_dataset.iloc[:, 0]
# print(train_labels)

# Convert videos to images for both train and test data
########### Uncomment below code if main data is lost ###########################
train_video_file = "data/Train Tom and jerry.mp4"
# convertVideoToImages(videoFile=train_video_file, directoryType="train", filename=train_filenames, framerate=framerate)

test_video_file = "data/Test Tom and jerry.mp4"
# convertVideoToImages(videoFile=test_video_file, directoryType="test", filename=test_filenames, framerate=framerate)

# initialize lists for storing augmented training data
augmented_train_images = []
augmented_train_labels = []

########### Uncomment below code if augmented data is lost ###########################
# # Make data augmentation as the number of samples available are limited
# count = 0
# shutil.rmtree("data/augmented-images", ignore_errors=True)
# for i in train_filenames:
#     train_images_loc = "data/train-images/" + str(i)
#     if not os.path.exists("data/augmented-images/frame{}".format(count)):
#         os.makedirs("data/augmented-images/frame{}".format(count))
#     augmented_train_images_loc = "data/augmented-images/frame" + str(count)
#     shutil.copy(train_images_loc, augmented_train_images_loc)
#     augmentData(img_file=train_images_loc, noOfNewFiles=9, saveDir=augmented_train_images_loc, imageName=str(i))
#     # for filename in os.listdir(augmented_train_images_loc):
#     #     img = cv.imread(os.path.join(augmented_train_images_loc, filename), 0)
#     #     img = cv.resize(img, (128, 128))
#     #     if img is not None:
#     #         augmented_train_images.append(img)
#     #         augmented_train_labels.append(train_labels[count])
#     print("Data loaded for frame" + str(count))
#     count += 1

# Load the augmented data into empty lists
k = 0
for folder in os.listdir("data/augmented-images"):
    for filename in os.listdir(os.path.join("data/augmented-images/") + folder):
        # print(filename)
        img = cv.imread(os.path.join("data/augmented-images/" + str(folder), filename), 0)
        img = cv.resize(img, (28, 28))
        if img is not None:
            augmented_train_images.append(img)
            #print(train_labels[k])
            augmented_train_labels.append(categorical_train_labels[k])
    print("Data loaded for " + str(folder))
    k += 1
    if not os.path.exists("data/augmented-images/frame{}".format(k)):
        break

# Load the data into empty list
temp_test_images = []
for filename in os.listdir("data/test-images"):
    img = cv.imread(os.path.join("data/test-images/", filename), 0)
    img = cv.resize(img, (28, 28))
    if img is not None:
        temp_test_images.append(img)

# Convert the lists to numpy arrays for further processsing
augmented_train_images = np.array(augmented_train_images)
augmented_train_images = augmented_train_images / 255.0
train_labels = np.array(augmented_train_labels)
temp_test_images = np.array(temp_test_images)
temp_test_images = temp_test_images / 255.0

# Reshape the data to fit into the model
train_images = augmented_train_images.reshape(len(augmented_train_images), 28, 28, 1)
test_images = temp_test_images.reshape(len(temp_test_images), 28, 28, 1)

# print(train_labels[0:50])
# exit(0)

##################  Training part  #######################
# For Hyper-parameter optimization
def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(
        filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=16),
        kernel_size=hp.Choice('conv_1_kernel', values=[3, 5]),
        activation='relu',
        input_shape=(28, 28, 1)
    ))
    model.add(keras.layers.Conv2D(
        filters=hp.Int('conv_2_filter', min_value=32, max_value=64, step=16),
        kernel_size=hp.Choice('conv_2_kernel', values=[3, 5]),
        activation='relu'
    ))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(
        units=hp.Int('dense_1_units', min_value=32, max_value=128, step=16),
        activation='relu'
    ))
    model.add(keras.layers.Dense(5, activation='softmax'))

    model.compile(
        optimizer=keras.optimizers.Adagrad(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model


tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    directory='output',
    project_name='new'
)

tuner.search(train_images, train_labels, epochs=3, validation_split=0.1)

model = tuner.get_best_models(num_models=1)[0]

model.summary()

model.fit(train_images, train_labels, batch_size=100 , epochs=10, validation_split=0.1, initial_epoch=3)

i=0
img = test_images[i]
img = np.expand_dims(img, axis=0)
prediction = model.predict_classes(img)
plt.imshow(temp_test_images[0])
plt.show()
print(prediction)

#################### Just for printing things ##########################

# print(train_images.shape)
# print(train_labels.shape)
# print(test_images.shape)
# print(test_images[3])
# plt.imshow(temp_test_images[3])
# plt.show()
