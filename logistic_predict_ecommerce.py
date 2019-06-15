import numpy as np 
from ecommerce_process import get_binary_data


X, Y = get_binary_data()

D = X.shape[1]
W = np.random.randn(D)
b = 0


def sigmoid(a):
	return 1/(1 + np.exp(-a))

def forward(X, W, b):
	return sigmoid(X.dot(W) + b)


P_Y_given_x = forward(X, W, b)
predictions = np.round(P_Y_given_x)


def classification_rate(Y, P):
	return np.mean(Y == P)

print("Score:", classification_rate(Y, predictions))