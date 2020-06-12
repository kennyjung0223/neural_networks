import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('mobile_price_data/train.csv')

X = dataset.iloc[:,:20].values

y = dataset.iloc[:,20:21].values

# normalize the data to account for dataset features
# normalization prevents gradients changing differently for every column
sc = StandardScaler()
X = sc.fit_transform(X)

# one hot encoding
# used to create unique binary values for each class
ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()

# training data will have 90% samples, test data will have 10%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# neural network
model = keras.Sequential()
model.add(layers.Dense(16, input_dim=20, activation='relu'))
model.add(layers.Dense(12, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',
			  metrics=['accuracy'])

# training model
history = model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=100, batch_size=64)


# checking model's performance
y_pred = model.predict(X_test)

# converting predictions to label
pred = list()
for i in range(len(y_pred)):
	pred.append(np.argmax(y_pred[i]))

# converting one hot encoded test label to label
test = list()
for i in range(len(y_test)):
	test.append(np.argmax(y_test[i]))

a = accuracy_score(pred, test)
print('Accuracy is:', a*100)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

