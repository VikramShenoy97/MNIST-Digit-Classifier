import numpy as np
import keras
import matplotlib.pyplot as plt
import random
from keras.models import Model, load_model
import cv2
from keras.datasets import mnist
from keras.utils import np_utils
from draw_graph import drawGraph
from sklearn.metrics import confusion_matrix

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
test_labels = test_labels.reshape(test_labels.shape[0], 1)
test_images = test_images / 255.
test_labels = np_utils.to_categorical(test_labels)
n_classes = test_labels.shape[1]

loaded_model = load_model("Model.h5")
loaded_model.set_weights(loaded_model.get_weights())

print loaded_model.summary()

score = loaded_model.evaluate(test_images, test_labels, verbose=1)
predictions = loaded_model.predict(test_images, verbose=0)
print score

y_pred = np.argmax(predictions, axis=1)
y_test = np.argmax(test_labels, axis=1)
labels = np.unique(y_test)

confusion_matrix = confusion_matrix(y_test, y_pred)

test_images = test_images.reshape(test_images.shape[0], 28, 28)
test_images = test_images * 255.
sample_indices = random.sample(range(test_images.shape[0]), 10)
sample_images = [test_images[i] for i in sample_indices]
sample_labels = [np.argmax(test_labels[i]) for i in sample_indices]
predicted = [np.argmax(predictions[i]) for i in sample_indices]

print sample_labels
print predicted
print confusion_matrix
fig = plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    prediction = predicted[i]
    plt.subplot(5, 2,1+i)
    plt.axis('off')
    color='green' if truth == prediction else 'red'
    plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction),
             fontsize=12, color=color)
    plt.imshow(sample_images[i])

plt.show()
drawGraph(confusion_matrix, labels)
