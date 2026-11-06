from lse import least_squares_estimation
import numpy as np
# import tqdm
# import scipy.spatial as ss



def ransac_estimator(X1, X2, num_iterations=60000):
    """
    Estimates the essential matrix using RANSAC algorithm.

    Inputs:
    - X1: Nx3 Matrix, Calibrated points from first camera
    - X2: Nx3 Matrix, Calibrated points from second camera
    - num_iterations: int, Number of RANSAC iterations (default: 60000)

    Outputs:
    - E: 3x3 Matrix, Best essential matrix found
    - inliers: ndarray of shape (n,), Indices of inlier matches
    """
    sample_size = 8

    eps = 1e-4

    best_num_inliers = -1
    best_inliers = None
    best_E = None

    for i in range(num_iterations):
        # permuted_indices = np.random.permutation(np.arange(X1.shape[0]))
        permuted_indices = np.random.RandomState(seed=(i*10)).permutation(np.arange(X1.shape[0]))
        sample_indices = permuted_indices[:sample_size]
        test_indices = permuted_indices[sample_size:]

        """
        E: ndarray of shape (3,3)
            Essential matrix.
        inliers: ndarray of shape (n,)
            Indices of inlier matches.
        """
        ##### STUDENT CODE START #####
        X1 = np.asarray(X1)
        X2 = np.asarray(X2)
        X1_8 = X1[sample_indices]
        X2_8 = X2[sample_indices]

        E = least_squares_estimation(X1_8, X2_8)
        inlier_indices = []
        e3_skew = [[0, -1, 0],[1, 0, 0],[0,0,0]]

        for j, x1_test in enumerate(X1):
            if j in sample_indices:
                continue

            x2_test = X2[j]
            Ex1  = x1_test @ E.T
            Etx2 = x2_test @ E

            num = np.sum(x2_test * Ex1) ** 2
            d_x2_x1 = num / (Ex1[0]**2  + Ex1[1]**2)
            d_x1_x2 = num / (Etx2[0]**2 + Etx2[1]**2)
            
            if (d_x2_x1 + d_x1_x2) < 10**-4:
                inlier_indices.append(j)
        inliers = np.asarray(inlier_indices)

        ##### STUDENT CODE END #####

        if inliers.shape[0] > best_num_inliers:
            best_num_inliers = inliers.shape[0]
            best_E = E
            best_inliers = inliers


    return best_E, best_inliers

if __name__ == "__main__":
    T = np.array([0.9, 2, 1.5])
    Tx = np.array([[0, -T[2], T[1]], [T[2], 0, -T[0]], [-T[1], T[0], 0]])
    Rot = ss.transform.Rotation.from_euler('xyz', [0.5, 0.3, 0.2]).as_matrix()
    E = Tx.dot(Rot)
    q = np.random.rand(9, 3)
    q[:, 0] /= q[:, 2]
    q[:, 1] /= q[:, 2]
    q[:, 2] /= q[:, 2]
    L = q.dot(E)
    p = np.random.rand(9, 3)
    p[:, 0] /= p[:, 2]
    p[:, 1] /= p[:, 2]
    p[:, 2] /= p[:, 2]
    p[:, 0] = (-L[:, 2]-L[:, 1]*p[:, 1])/L[:, 0]
    ransac_estimator(p, q)
