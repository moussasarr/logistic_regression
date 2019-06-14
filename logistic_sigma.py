# Basic logistic regression
# Applying sigma sigmoid function

import numpy as np 

N = 100 
D = 2
X = np.random.randn(N, D)
ones = np.array([[1]*N]).T
Xb = np.hstack((ones, X))
W = np.random.randn(D + 1)
Z = Xb.dot(W)

def sigmoid(Z):
	return 1/(1 + np.exp(-Z))

print(sigmoid(Z))
