# the hello world of machine learning
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

iris = load_iris()

print("Example data: ")
print(iris.data[:5])
print("Example labels: ")
print(iris.target[:5])

X = iris.data
y = iris.target.reshape(-1, 1)

encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = keras.Sequential()
model.add(layers.Dense(10, input_shape=(4,), activation="relu"))
model.add(layers.Dense(10, activation="relu"))
model.add(layers.Dense(3, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

print("Neural Network Model Summary: ")
print(model.summary())

model.fit(X_train, y_train, verbose=2, batch_size=5, epochs=200)

results = model.evaluate(X_test, y_test)

print("Final test set loss: {:4f}".format(results[0]))
print("Final test set accuracy: {:4f}".format(results[1]))


