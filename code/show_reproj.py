import numpy as np
import matplotlib.pyplot as plt

def show_reprojections(image1, image2, uncalibrated_1, uncalibrated_2, P1, P2, K, T, R, plot=True):
	"""
	Reprojects 3D points back to both images to visualize reconstruction accuracy.

	Inputs:
	- image1: HxWx3 Matrix, First image
	- image2: HxWx3 Matrix, Second image
	- uncalibrated_1: 2xN or 3xN Matrix, Original keypoints from image1
	- uncalibrated_2: 2xN or 3xN Matrix, Original keypoints from image2
	- P1: Nx3 Matrix, 3D points in camera 1 coordinate frame
	- P2: Nx3 Matrix, 3D points in camera 2 coordinate frame
	- K: 3x3 Matrix, Intrinsic camera matrix
	- T: 3x1 Vector, Translation vector
	- R: 3x3 Matrix, Rotation matrix
	- plot: Boolean, If True displays images with reprojections; if False returns projection matrices

	Outputs (if plot=False):
	- P1proj: Nx3 Matrix, Homogeneous coordinates of P1 projected to image2
	- P2proj: Nx3 Matrix, Homogeneous coordinates of P2 projected to image1
	"""

	##### STUDENT CODE START #####
	T = np.asarray(T).reshape(3, 1)

	# cam-1 points -> cam 2's frame (P2 = R P1 + T) -> pixels
	P1_in_cam2 = R @ P1.T + T # 3 x N
	P1proj = (K @ P1_in_cam2).T # N x 3

	# cam-2 points -> cam 1's frame (P1 = R^T (P2 - T)) -> pixels
	P2_in_cam1 = R.T @ (P2.T - T) # 3 x N
	P2proj = (K @ P2_in_cam1).T # N x 3
	##### STUDENT CODE END #####

	if (plot):
		plt.figure(figsize=(6.4*3, 4.8*3))
		ax = plt.subplot(1, 2, 1)
		ax.set_xlim([0, image1.shape[1]])
		ax.set_ylim([image1.shape[0], 0])
		plt.imshow(image1[:, :, ::-1])
		plt.plot(P2proj[:, 0] / P2proj[:, 2],
			P2proj[:, 1] / P2proj[:, 2], 'bs')
		plt.plot(uncalibrated_1[0, :], uncalibrated_1[1, :], 'ro')

		ax = plt.subplot(1, 2, 2)
		ax.set_xlim([0, image1.shape[1]])
		ax.set_ylim([image1.shape[0], 0])
		plt.imshow(image2[:, :, ::-1])
		plt.plot(P1proj[:, 0] / P1proj[:, 2],
			P1proj[:, 1] / P1proj[:, 2], 'bs')
		plt.plot(uncalibrated_2[0, :], uncalibrated_2[1, :], 'ro')
		plt.savefig('result/show_reproj.png', bbox_inches='tight', dpi=150)

	else:
		return P1proj, P2proj
