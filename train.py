import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.datasets import mnist
from keras.utils import np_utils

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
train_labels = train_labels.reshape(train_labels.shape[0], 1)

train_images = train_images / 255.

train_labels = np_utils.to_categorical(train_labels)
n_classes = train_labels.shape[1]


model = Sequential()
model.add(Conv2D(6, strides=1, kernel_size=5, activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=[2,2], strides=2))
model.add(Dropout(0.25))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=[2,2], strides=2))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(84, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
pd.DataFrame(model.fit(train_images, train_labels, epochs=15, validation_split=0.1, verbose=1).history).to_csv("history.csv")
model.save("Model.h5")
