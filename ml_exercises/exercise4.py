# -*- coding: utf-8 -*-
"""SHEET 4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13GDV3dXAiQKwKvHmXX6J1CEdDiNaokNr
"""

# Commented out IPython magic to ensure Python compatibility.
import tensorflow as tf
import keras
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
# %matplotlib inline
import matplotlib.pyplot as plt
tf.random.set_seed(0)
import numpy as np
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l1

"""TASK 3: OPTIMIZING NEURAL NETWORK TRAINING

---


"""

(trainX, trainy), (testX, testy) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
#https://mrdatascience.com/how-to-plot-mnist-digits-using-matplotlib/

num = 4
images = trainX[:num]
labels = trainy[:num]

num_row = 2
num_col = 2 # plot images
fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
for i in range(num):
    ax = axes[i//num_col, i%num_col]
    ax.imshow(images[i], cmap='gray')
    ax.set_title('Label: {}'.format(labels[i]))
plt.tight_layout()
plt.show()

def prep_pixels(train, test):
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	return train_norm, test_norm

X_train, X_test = prep_pixels(trainX, testX)

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
trainy = tf.keras.utils.to_categorical(trainy, 10)
testy = tf.keras.utils.to_categorical(testy, 10)

model = Sequential()
model.add(Flatten(input_shape=(784, )))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

for l in model.layers:
    print(l.name, l.input_shape,'==>',l.output_shape)

epochs = 5
opt = tf.keras.optimizers.Adam()
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

history1 = model.fit(X_train, trainy, epochs = epochs)
history1

score = model.evaluate(X_test, testy)
print('Test loss:', round(score[0], 3))
print('Test accuracy:', round(score[1], 3))

model = Sequential()
model.add(Flatten(input_shape=(784, )))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

for l in model.layers:
    print(l.name, l.input_shape,'==>',l.output_shape)

epochs = 5
opt = tf.keras.optimizers.Adam()
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

history2 = model.fit(X_train, trainy, epochs = epochs)
history2

score = model.evaluate(X_test, testy)
print('Test loss:', round(score[0], 3))
print('Test accuracy:', round(score[1], 3))

plt.figure(figsize=(11, 6), dpi=80)
plt.plot(history1.history['loss'], label = "loss first model")
plt.plot(history1.history['accuracy'], label = "accuracy first model")
plt.plot(history2.history['loss'], label = "loss second model")
plt.plot(history2.history['accuracy'], label = "accuracy second model")
plt.title('model loss')
plt.ylabel('loss & accuracy')
plt.xlabel('epoch')
plt.legend(loc='center left', bbox_to_anchor=(0.5, 0.5))
plt.show()

"""TASK 4: SOLVING TRAINING PROBLEMS"""

(trainX, trainy), (testX, testy) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

X_train, X_test = prep_pixels(trainX, testX)

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
trainy = tf.keras.utils.to_categorical(trainy, 10)
testy = tf.keras.utils.to_categorical(testy, 10)

model = Sequential()
model.add(Flatten(input_shape=(784, )))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))

epochs = 5
opt = tf.keras.optimizers.Adam()
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

history3 = model.fit(X_train, trainy, epochs = epochs)
history3

score = model.evaluate(X_test, testy)
print('Test loss:', round(score[0], 3))
print('Test accuracy:', round(score[1], 3))

model = Sequential()
model.add(Flatten(input_shape=(784, )))
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

epochs = 5
opt = tf.keras.optimizers.Adam()
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

history4 = model.fit(X_train, trainy, epochs = epochs)
history4

score = model.evaluate(X_test, testy)
print('Test loss:', round(score[0], 3))
print('Test accuracy:', round(score[1], 3))

model = Sequential()
model.add(Flatten(input_shape=(784, )))
model.add(Dense(256, activation='relu', kernel_regularizer='l1'))
model.add(Dense(128, activation='relu', kernel_regularizer='l1'))
model.add(Dense(64, activation='relu', kernel_regularizer='l1'))
model.add(Dense(10, activation='softmax'))

epochs = 5
opt = tf.keras.optimizers.Adam()
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

history5 = model.fit(X_train, trainy, epochs = epochs)
history5

score = model.evaluate(X_test, testy)
print('Test loss:', round(score[0], 3))
print('Test accuracy:', round(score[1], 3))

plt.figure(figsize=(11, 6), dpi=80)
plt.plot(history3.history['loss'], label = "loss first model")
plt.plot(history3.history['accuracy'], label = "accuracy first model")
plt.plot(history4.history['loss'], label = "loss second model")
plt.plot(history4.history['accuracy'], label = "accuracy second model")
plt.plot(history5.history['loss'], label = "loss third model")
plt.plot(history5.history['accuracy'], label = "accuracy third model")
plt.title('model loss')
plt.ylabel('loss & accuracy')
plt.xlabel('epoch')
plt.legend(loc='upper left', bbox_to_anchor=(0.8, 0.5))
plt.show()