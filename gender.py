import numpy as np
import pandas as pd
import os
import cv2 as cv
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
from keras.models import Sequential

gender = pd.read_csv('../input/face-gender/gender.csv', index_col=0)
base = "../input/faceproj/resized_faces/resized_faces/"
a = []
for i in gender.index:
    img = cv.imread(base+i, 0)
    a.append(img.tolist())
X = np.array(a).reshape(-1, 64, 64, 1)
X = np.array(X / 255)
del a

y = np.array(gender).flatten()
X, y = shuffle(X, y)
X_train, y_train = X, y

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=10, 
    horizontal_flip=True)
datagen.fit(X_train)

model = Sequential()
model.add(Conv2D(32, kernel_size=(5,5), activation='relu', input_shape=X.shape[1:]))
model.add(Conv2D(32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(64, kernel_size=(5,5), activation='relu'))
model.add(Conv2D(64, kernel_size=(5,5), activation='relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit_generator(datagen.flow(X_train, y_train, batch_size=64), steps_per_epoch=len(X_train) / 64, epochs=30)

model.save('gender_model.h5')
