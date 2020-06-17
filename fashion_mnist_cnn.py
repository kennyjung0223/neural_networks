import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

"""
In reference to

https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb
https://www.tensorflow.org/tutorials/keras/classification

"""
if __name__ == "__main__":
	fashion_mnist = keras.datasets.fashion_mnist

	(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

	class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
				   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

	"""
	plt.figure()
	plt.imshow(train_images[0])
	plt.colorbar()
	plt.grid(False)
	plt.show()

	plt.figure(figsize=(10, 10))
	for i in range(25):
		plt.subplot(5, 5, i + 1)
		plt.xticks([])
		plt.yticks([])
		plt.grid(False)
		plt.imshow(train_images[i], cmap=plt.cm.binary)
		plt.xlabel(class_names[train_labels[i]])
	plt.show()

	"""

	train_images = train_images.reshape((60000, 28, 28, 1))
	train_images = train_images / 255.0

	test_images = test_images.reshape((10000, 28, 28, 1))
	test_images = test_images / 255.0

	# building the model
	model = Sequential()

	model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(64, (5, 5), activation='relu'))
	model.add(layers.MaxPooling2D((2, 2)))

	"""model.add(layers.Conv2D(64, (7, 7), activation='relu', padding='same', input_shape=(28, 28, 1)))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(layers.MaxPooling2D((2, 2)))"""

	model.add(layers.Flatten())
	model.add(layers.Dense(64, activation='relu'))
	model.add(layers.Dense(10, activation='softmax'))

	model.compile(loss='sparse_categorical_crossentropy',
				  optimizer='adam',
				  metrics=['accuracy'])

	model.fit(train_images, train_labels, epochs=5)
	test_loss, test_acc = model.evaluate(test_images, test_labels)

	print("Test accuracy:", test_acc)

	predictions = model.predict(test_images)

	train_images = train_images.reshape((60000, 28, 28))
	test_images = test_images.reshape((10000, 28, 28))

	print(f"Prediction for {class_names[test_images[0]]}: {np.argmax(predictions[0])}")

	num_rows, num_cols = 7, 2
	num_images = num_rows * num_cols
	plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))

	for i in range(num_images):
		plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
		plot_image(i, predictions[i], test_labels, test_images)
		plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
		plot_value_array(i, predictions[i], test_labels)

	plt.tight_layout()
	plt.show()










