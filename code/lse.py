import numpy as np

def least_squares_estimation(X1, X2):
	""" 
	inputs:
	X1: Nx3 Matrix Calibrated Point
	X2: Nx3 Matrix Calibrated Point
	X1 ~ X2

	rets:
	E: Essential Matrix using 8 point algorithm
	"""
	##### STUDENT CODE START #####
	#set up A
	N = len(X1)
	A = np.empty((N,9))

	for n in range(N):
		pr = X1[n]
		pl = X2[n]
		for i in range(3):
			for j in range(3):
				A[n][3*i+j] = pl[i] * pr[j] 
	
	#2
	_, _, Vt = np.linalg.svd(A)
	v9 = Vt[8] # 1x9
	E = v9.reshape(3,3)	#3x3

	#3
	U, _, Vt = np.linalg.svd(E)
	S = np.diag([1, 1, 0])
	return U @ S @ Vt

	##### STUDENT CODE END #####
	return E
