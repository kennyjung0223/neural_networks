import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

"""
A basic convolutional neural network with 1 stride, 0 padding, and max-pooling

We will use 32 filters, 5 x 5 window for the convolutional layer, 2 x 2 window for pooling layer
ReLU activation function
input tensor is size (28, 28, 1)
"""

model = Sequential()
model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))

# number of parameters in conv2D layer = 32 * (25 + 1) = 832
# 32 * (5 * 5 + 1) where 1 is the bias
# max pooling requires 0 parameters since it is a mathematical operation

# model.summary()

# second group of layers with 64 filters, 5 x 5 conv window, and 2 x 2 pooling layer
# now ((5 * 5 * 32) + 1) * 64 = 51264

model.add(layers.Conv2D(64, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# flatten so our 3D tensor becomes a 1D tensor
model.add(layers.Flatten())
model.add(layers.Dense(10, activation='softmax'))

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# print(train_images.shape)	# (60000, 28, 28)
# print(train_labels.shape)	# (60000,)

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.compile(loss='categorical_crossentropy', 
			  optimizer='sgd', 
			  metrics=['accuracy'])

model.fit(train_images, train_labels, batch_size=100, epochs=5, verbose=1)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)








