import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

"""
# When to use a Sequential Model

# Define Sequential model with 3 layers
model = keras.Sequential(
	[
		layers.Dense(2, activation="relu", name="layer1"),
		layers.Dense(3, activation="relu", name="layer2"),
		layers.Dense(4, name="layer3"),
	]
)
# Call model on a test input
x = tf.ones((3, 3))
y = model(x)

# is equivalent to

# Create 3 layers
layer1 = layers.Dense(2, activation="relu", name="layer1")
layer2 = layers.Dense(3, activation="relu", name="layer2")
layer3 = layers.Dense(4, name="layer3")

# Call layers on a test input
x = tf.ones((3, 3))
y = layer3(layer2(layer1(x)))
"""

"""
# Creating a Sequential Model

model = keras.Sequential(
	[
		layers.Dense(2, activation="relu"),
		layers.Dense(3, activation="relu"),
		layers.Dense(4),
	]
)

print(model.layers)

# can also be created incrementally via add() method

model = keras.Sequential()
model.add(layers.Dense(2, activation="relu"))
model.add(layers.Dense(3, activation="relu"))
model.add(layers.Dense(4))

print(model.layers)

# can also pop like lists
model.pop()
print(len(model.layers)) # 2
"""

"""
# Specifying the input shape in advance

# initialized layer has no weights unless called on an input

layer = layers.Dense(3)
layer.weights # Empty

# Call layer on a test input
x = tf.ones((1, 4))
y = layer(x)
print(layer.weights) # now it has weights of shape (4, 3) and (3,)

# Same thing applies to Sequential models

model = keras.Sequential(
	[
		layers.Dense(2, activation="relu"),
		layers.Dense(3, activation="relu"),
		layers.Dense(4),
	]
)	# No weights at this stage

# At this point, you can't do this:
# model.weights

# You also can't do this
# model.summary()

# Call the model on a test input
x = tf.ones((1, 4))
y = model(x)
print("Number of weights after calling the model:", len(model.weights)) # 6

model.summary()



# however, can be very helpful to build incrementally
# so that we can display the summary of the model so far
# start by passing an Input object

model = keras.Sequential()
model.add(keras.Input(shape=(4,)))
model.add(layers.Dense(2, activation="relu"))
model.summary()

# input object is not displayed as part of model.layers though
print(model.layers)

# simple alternative is to pass an input_shape argument to the first layer
model = keras.Sequential()
model.add(layers.Dense(2, activation="relu", input_shape=(4,)))
model.summary()

"""

"""
# A common debugging workflow: add() + summary()

model = keras.Sequential()
model.add(keras.Input(shape=(250, 250, 3))) # 250x250 RGB images
model.add(layers.Conv2D(32, 5, strides = 2, activation="relu"))
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.MaxPooling2D(3))

# Can you guess what the current ouput shape is at this point? Probably not
# Let's just print it:
model.summary()

# answer was (40, 40, 32), so we can keep downsampling

model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.MaxPooling2D(3))
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.MaxPooling2D(2))

# and now
model.summary()

# Now that we have a 4x4 feature maps, time to apply global max pooling
model.add(layers.GlobalMaxPooling2D())

# Finally, we add a classification layer
model.add(layers.Dense(10))

"""

"""
# What to do once you have a model

# 1). Train model, evaluate it, and run inference
# 2). Save model to disk and restore it
# 3). Speed up model training by leveraging multiple GPUs

"""

"""
# Feature extraction with a sequential model

initial_model = keras.Sequential(
	[
		keras.Input(shape=(250, 250, 3)),
		layers.Conv2D(32, 5, strides=2, activation="relu"),
		layers.Conv2D(32, 3, activation="relu"),
		layers.Conv2D(32, 3, activation="relu"),
	]
)
feature_extractor = keras.Model(
	inputs=initial_model.inputs,
	outputs=[layer.output for layer in initial_model.layers],
)

# Call feature extractor on test input
x = tf.ones((1, 250, 250, 3))
features = feature_extractor(x)

# similar example with only extract features from one layer
initial_model = keras.Sequential(
	[
		keras.Input(shape=(250, 250, 3)),
		layers.Conv2D(32, 5, strides=2, activation="relu"),
		layers.Conv2D(32, 3, activation="relu", name="my_intermediate_layer"),
		layers.Conv2D(32, 3, activation="relu"),
	]
)
feature_extractor = keras.Model(
	inputs=initial_model.inputs,
	outputs=initial_model.get_layer(name="my_intermediate_layer").output,
)
# Call feature extractor on test input
x = tf.ones((1, 250, 250, 3))
features = feature_extractor(x)

"""

"""
# Transfer learning with a Sequntial model

"""
model = keras.Sequential([
	keras.Input(shape=(784)),
	layers.Dense(32, activation='relu'),
	layers.Dense(32, activation='relu'),
	layers.Dense(32, activation='relu'),
	layers.Dense(10),
])

# presumably you would want to first load pre-trained weights
model.load_weights(...)

# Freeze all layers except the last one
for layer in model.layers[:-1]:
	layer.trainable = False

# Recompile and train (this will only update the weights of the last layer)
model.compile(...)
model.fit(...)

# another common blueprint
# stack a pre-trained model and some freshly initialized classification layers

# load a convolutional base with pre-trained weights
base_model = keras.applications.Xception(
	weights='imagenet',
	include_top=False,
	pooling='avg')

# freeze the base model
base_model.trainable = False

# use a sequential model to add a trainable classifer on top
model = keras.Sequential([
	base_model,
	layers.Dense(1000),
])

# compile and train
model.compile(...)
model.fit(...)




