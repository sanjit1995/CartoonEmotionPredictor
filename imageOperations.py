import cv2  # for capturing videos
import math  # for mathematical operations
import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


def convertVideoToImages(videoFile, directoryType, filename, framerate):
    count = 0

    if not os.path.exists("data/{}-images/".format(directoryType)):
        os.makedirs("data/{}-images/".format(directoryType))

    fileLoc = str("data/" + str(directoryType) + "-images/")

    # capturing the video from the given path
    cap = cv2.VideoCapture(videoFile)

    # Given frame rate
    frameRate = cap.get(framerate)
    x = 1
    while cap.isOpened():
        # get current frame number
        frameId = cap.get(1)
        ret, frame = cap.read()
        if not ret:
            break
        if frameId % math.floor(frameRate) == 0:
            file = filename[count]
            count += 1
            cv2.imwrite(fileLoc + file, frame)
    cap.release()
    print("Done!")


# convertVideoToImages("data/Train Tom and jerry.mp4", "train")

def augmentData(img_file, noOfNewFiles, saveDir, imageName):

    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    img = load_img(img_file)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)

    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=saveDir, save_prefix=imageName, save_format='jpg'):
        i += 1
        if i >= noOfNewFiles:
            break


# augmentData(img_file="data/train-images/frame0.jpg", noOfNewFiles=9, saveDir="frame0",imageName="frame0")
