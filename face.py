import numpy as np
import pandas as pd
import os
import cv2 as cv
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout


# set base as the folder path of face images
base = "../input/resized_faces/resized_faces/"
face_list = os.listdir(base)
a = []
for i in face_list:
    img = cv.imread(base+i, 0)
    temp = img.tolist()
    a.append(temp)
face = np.array(a)
del a

# set base as the folder path of non-face images
base = "../input/valid_1/valid_1/"
nface_list = os.listdir(base)
a = []
for i in nface_list:
    img = cv.imread(base+i, 0)
    temp = img.tolist()
    a.append(temp)
nonface = np.array(a)
del a

data = np.concatenate((face, nonface), axis=0)
y = np.zeros(len(face)+len(nonface))
for i in range(len(face)):
    y[i] = 1
	
X, y = shuffle(data, y)
X = X.reshape(-1, 64, 64, 1)
X = X / 255.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=10)
datagen.fit(X)

model = Sequential()
model.add(Conv2D(20, kernel_size=(5,5), input_shape=X.shape[1:], activation='relu'))
model.add(Conv2D(40, kernel_size=(7,7), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(80, kernel_size=(9,9), activation='relu'))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')

model.fit_generator(datagen.flow(X_train, y_train, batch_size=64), steps_per_epoch=len(X) / 64, epochs=25)

model.save('face_model.h5')