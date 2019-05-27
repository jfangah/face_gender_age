import numpy as np
import pandas as pd
import os
import cv2 as cv
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D


def age_mask(age):
    y_age = np.zeros(len(age))
    flag = 1
    for i in range(len(age)):
        if age[i] == 1:
            flag = 0
            return y_age
        if flag == 1:
            y_age[i] = 1
            continue

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
y = []
for i in face_list:
    y.append(i[-6:-4])

y = np.array(y).astype(int)
y = y.flatten().tolist()
y = np.array(pd.get_dummies(y))
for i in range(len(y)):
    y[i] = age_mask(y[i])
X = face.reshape(-1, 64, 64, 1)
X = np.array(X / 255.0)

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=10,
    horizontal_flip=True)
datagen.fit(X_train)

model = Sequential()
model.add(Conv2D(20, kernel_size=(5,5), input_shape=X.shape[1:], activation='relu'))
model.add(Conv2D(40, kernel_size=(7,7), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(80, kernel_size=(9,9), activation='relu'))
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(79, activation='sigmoid'))

model.compile(optimizer='adam', loss='mse', metrics=['mse'])
model.fit_generator(datagen.flow(X_train, y_train, batch_size=64),steps_per_epoch=len(X_train) / 64, epochs=95)

model.save('age_model.h5')