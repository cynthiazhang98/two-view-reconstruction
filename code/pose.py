import numpy as np

def pose_candidates_from_E(E):
	""" 
	Computes four possible camera poses (rotation and translation) from the essential matrix.

    Inputs:
    - E: 3x3 Matrix, Essential Matrix

    Outputs:
    - transform_candidates: List of 4 dictionaries, each containing:
        - "T": 3x1 Vector, Translation vector
        - "R": 3x3 Matrix, Rotation matrix
	"""
	##### STUDENT CODE START #####
	U, _, Vt = np.linalg.svd(E)
	if np.linalg.det(U) * np.linalg.det(Vt) < 0:
		Vt[2, :] *= -1
	print("det U:", np.linalg.det(U), " det Vt:", np.linalg.det(Vt))
	Rz = np.array([[0, -1, 0],[1,  0, 0],[0,  0, 1]])
	T = U[:, 2]
	R_a = U @ Rz.T @ Vt
	R_b = U @ Rz @ Vt
	transform_candidates = [
        {"T":  T, "R": R_a},
        {"T":  T, "R": R_b},
        {"T": -T, "R": R_a},
        {"T": -T, "R": R_b}, 
    ]
	
    ##### STUDENT CODE END #####
	
	return transform_candidates