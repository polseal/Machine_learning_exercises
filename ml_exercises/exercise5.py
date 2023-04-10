# -*- coding: utf-8 -*-
"""Copy of 5 sheet.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1JOgW054ildFlB3oW8rvDyGaSlJVe8rsE
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from matplotlib import pyplot
import keras
from keras import metrics
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, Reshape, Flatten, MaxPool2D
from tensorflow.keras.utils import to_categorical

"""TASK 1"""

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

def prep_pixels(train, test):
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	return train_norm, test_norm

X_train, X_test = prep_pixels(x_train, x_test)

num = 5

index = np.random.choice(X_train.shape[0], num, replace=False)

def conversion(k, X_train, index):
  for i in range(num):
    result = ndimage.convolve(X_train[index[i]], k)
    finalOutput = result.squeeze()
    fig, ax = plt.subplots(ncols=2)
    ax[0].set_title("original")
    ax[0].imshow(x_train[index[i]])
    ax[1].set_title("box blurred")
    ax[1].imshow(result)
    plt.show()

k = np.array([[-1,-1,-1],[2,2,2],[-1,-1,-1]])

conversion(k, X_train, index)

k = np.array([[-1,2,-1],[-1,2,-1],[-1,2,-1]])
conversion(k, X_train, index)

k = np.array([[-1,-1,2],[-1,2,-1],[2,-1,-1]])
conversion(k, X_train, index)

"""TASK 2"""

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

x_train = (x_train / 255)
x_test = (x_test / 255)

X_train = np.expand_dims(x_train, axis=3)
X_test = np.expand_dims(x_test, axis=3)

model = Sequential([
  Conv2D(128, kernel_size=(3,3), input_shape=(28, 28, 1)),
  Conv2D(64, kernel_size=(3,3)),
  Flatten(),
  Dense(64),
  Dense(10, activation='softmax'),
])

epochs = 5
model.compile(
  'adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

model.fit(
  X_train,
  to_categorical(y_train),
  epochs=5,
  validation_data=(X_test, to_categorical(y_test)),
)

model.evaluate(X_test, to_categorical(y_test))

score = model.evaluate(X_test, to_categorical(y_test))
print('Test loss:', round(score[0], 3))
print('Test accuracy:', round(score[1], 3))

model_1 = Sequential([
  Conv2D(128, kernel_size=(3,3), input_shape=(28, 28, 1)),
  MaxPool2D(pool_size=(2, 2)),
  Conv2D(64, kernel_size=(3,3)),
  MaxPool2D(pool_size=(2, 2)),
  Flatten(),
  Dense(64),
  Dense(10, activation='softmax'),
])

epochs = 5
model_1.compile(
  'adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

model_1.fit(
  X_train,
  to_categorical(y_train),
  epochs=5,
  validation_data=(X_test, to_categorical(y_test)),
)

score = model_1.evaluate(X_test, to_categorical(y_test))
print('Test loss:', round(score[0], 3))
print('Test accuracy:', round(score[1], 3))