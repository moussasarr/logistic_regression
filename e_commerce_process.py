import numpy as np 
import pandas as pd 

def get_data():
	df = pd.read_csv("./data/ecommerce_data.csv")

	# Transform into a numpy array
	data = df.values

    # Differentiating Y from X
	X = data[:, : -1]
	Y = data[:, -1]

	# Normalize the continuous inputs (continous columns of X)
	X[:, 1] = (X[:, 1] - X[:, 1].mean())/ (X[:, 1].std())
	X[:, 2] = (X[:, 2] - X[:, 2].mean())/ (X[:, 2].std())

    # Use a data matrix that takes into account the 4 category outputs
	N, D = X.shape
	X2 = np.zeros((N, D+3))
	X2[:,: D-1] = X[:,:D-1]
	
	for n in range(N):
		t = int(X[n, D-1])
		X2[n, t + D-1] = 1
		return X2, Y

def get_binary_data():
	X, Y = get_data()
	X2 = X[Y <= 1]
	Y2 = Y[Y <= 1]
	return X2, Y2