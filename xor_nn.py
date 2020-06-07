import numpy as np

def sigmoid(z):
	return 1.0/(1 + np.exp(-z))

def initialize_parameters(n_x, n_h, n_y):
	W1 = np.random.randn(n_h, n_x)
	b1 = np.zeros((n_h, 1))
	W2 = np.random.rand(n_y, n_h)
	b2 = np.zeros((n_y, 1))

	parameters = {
		"W1": W1,
		"b1": b1,
		"W2": W2,
		"b2": b2,
	}

	return parameters

def forward_prop(X, parameters):
	W1 = parameters["W1"]
	b1 = parameters["b1"]
	W2 = parameters["W2"]
	b2 = parameters["b2"]

	Z1 = np.dot(W1, X) + b1
	A1 = np.tanh(Z1)
	Z2 = np.dot(W2, X) + b2
	A2 = np.sigmoid(Z2)

	cache = {
		"A1": A1,
		"A2": A2,
	}

	return A2, cache

def calculate_cost(A2, Y):
	cost = -np.sum(np.multiply(Y, np.log(A2)) + np.multiply(1-Y, np.log(1-A2)))/m
	cost = np.squeeze(cost)

	return cost