import numpy as np

def reconstruct3D(transform_candidates, calibrated_1, calibrated_2):
	"""
	Selects the correct camera pose from 4 candidates by triangulating points.
	
	This function selects (T,R) among the 4 candidates transform_candidates
	such that all triangulated points are in front of both cameras.

	Inputs:
	- transform_candidates: List of 4 dictionaries, each containing "T" and "R"
	- calibrated_1: Nx3 Matrix, Calibrated points from first camera
	- calibrated_2: Nx3 Matrix, Calibrated points from second camera

	Outputs:
	- P1: Nx3 Matrix, 3D points in camera 1 coordinate frame
	- P2: Nx3 Matrix, 3D points in camera 2 coordinate frame
	- T: 3x1 Vector, Best translation vector
	- R: 3x3 Matrix, Best rotation matrix
	"""
	best_num_front = -1
	best_candidate = None
	best_lambdas = None
	for candidate in transform_candidates:
		R = candidate['R']
		T = candidate['T']

		##### STUDENT CODE START #####
		N = calibrated_1.shape[0]
		lambdas = np.zeros((2, N))

		for i in range(N):
			A = np.column_stack([R @ calibrated_1[i], -calibrated_2[i]])   # 3 x 2
			lambdas[:, i] = np.linalg.lstsq(A, -T)[0]
		##### STUDENT CODE END #####
		
		num_front = np.sum(np.logical_and(lambdas[0]>0, lambdas[1]>0))

		if num_front > best_num_front:
			best_num_front = num_front
			best_candidate = candidate
			best_lambdas = lambdas
			print("best", num_front, best_lambdas[0].shape)
		else:
			print("not best", num_front)


	# lambdas[0] = lambda1 (cam1 depths), lambdas[1] = lambda2 (cam2 depths)
	P1 = best_lambdas[0].reshape(-1, 1) * calibrated_1  # 3D points in cam1 frame
	P2 = best_lambdas[1].reshape(-1, 1) * calibrated_2  # 3D points in cam2 frame
	T = best_candidate['T']
	R = best_candidate['R']
	return P1, P2, T, R
