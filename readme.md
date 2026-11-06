# Two-View 3D Reconstruction — Franka Kitchen

Recovering camera motion and 3D scene structure from two images, using epipolar geometry implemented from scratch in Python.

Originally the coding portion of Homework 3 for CIS 5800 (Machine Perception) at the University of Pennsylvania, Fall 2026.

## Overview

Two frames were captured by a ZED 2 stereo camera (DROID setup) while a robot moved through the Franka Kitchen at the GRASP Lab. The goal is to use only those two images to:

1. Recover the relative camera motion between the two views, a rotation `R` and translation `T`, such that a scene point at `P1` in camera 1 appears at `P2 = R·P1 + T` in camera 2.
2. Reconstruct the 3D positions of the matched scene points, producing a small 3D model of the workspace.

The camera is calibrated and the images are rectified (no radial distortion). Both views share the intrinsic matrix

```
K = [ f  0  u0 ]      f = 967,  (u0, v0) = (640, 360)
    [ 0  f  v0 ]
    [ 0  0   1 ]
```

## Pipeline

```
two images
   │  SIFT keypoints + brute-force matching (OpenCV, provided)
   ▼
noisy pixel correspondences  (contain outliers)
   │  K⁻¹ converts pixels u to calibrated coordinates x
   ▼
8-point algorithm + RANSAC  ──►  essential matrix E
   │  SVD decomposition of E
   ▼
4 candidate poses (R, T)
   │  triangulate points, keep the pose with most points in front of both cameras
   ▼
camera motion (R, T) + 3D points
   │  reproject and compare against the original images
   ▼
validation
```

## What was implemented

Six modules in `code/`. SIFT matching, the notebook, and the visualizer were provided with the course; the geometry below is the implemented part.

| File | What it does |
|---|---|
| `lse.py` | **Essential matrix from matches (8-point algorithm).** Builds the epipolar constraint `x2ᵀ E x1 = 0` for N matched calibrated points, solves for `E` by SVD, then projects it onto the space of valid essential matrices by replacing its singular values with `(1, 1, 0)`. |
| `ransac.py` | **Robust estimation with RANSAC.** SIFT matches contain outliers, so `E` is repeatedly estimated from random sets of 8 correspondences. Each estimate is scored by the symmetric epipolar distance of every other match, and the estimate with the largest consensus set (residual below ε = 10⁻⁴) is kept. The iteration count follows from the desired failure probability. |
| `plot_epi.py` | **Epipolar lines.** Converts `E` to the fundamental matrix `F = K⁻ᵀ E K⁻¹` and draws, in image 2, the line on which each image-1 point must lie. If `E` and `K` are right, the lines pass through the matching points. |
| `pose.py` | **Pose candidates from E.** SVD of `E` gives the twisted-pair ambiguity (two rotations and two translation signs), and the sign ambiguity of `E` vs. `−E` gives four candidate `(R, T)` pairs in total. |
| `recon3d.py` | **Triangulation and pose selection.** For each candidate, solves the least-squares problem `λ2·x2 = λ1·R·x1 + T` for the depths of every match (the two rays rarely intersect exactly). The correct pose is the one that puts the most points in front of both cameras (`λ1, λ2 > 0`). |
| `show_reproj.py` | **Reprojection check.** Projects the reconstructed points into the other camera and overlays them on the image, as a visual test of the recovered `(R, T)`. |

## Key ideas

- **Epipolar geometry.** A point in one image constrains its match in the other image to a line. The essential matrix `E = [T]× R` encodes that constraint for calibrated cameras, so estimating `E` from matches is how relative motion is recovered.
- **Why RANSAC.** The 8-point algorithm is a least-squares fit, so a few wrong SIFT matches can ruin the estimate. RANSAC fits many small samples and keeps the one that the most other matches agree with.
- **Twisted-pair ambiguity.** Decomposing `E` yields four mathematically valid `(R, T)` pairs, and only one places the scene in front of both cameras. The depth test resolves it.
- **Scale ambiguity.** From two views alone, depths and the translation are only known up to a global scale, so the reconstruction is in relative units.

## Results

The pipeline produces:

- RANSAC inlier matches between the two views (outliers rejected)
- Epipolar lines in image 2 passing through the corresponding points on the Franka arm
- A top-down (X, Z) view of the reconstructed points and both camera positions
- Reprojections of the 3D points that land on their image locations in both views

<!-- Add screenshots here, e.g. ![RANSAC inliers](figures/inliers.png) -->

## Running it

**Locally**

1. `python -m pip install -r requirements.txt`
2. Open `hw3.ipynb` from the `homework_3/` directory and run the cells in order.
3. Optional: run the interactive visualizer from `code/` with `python visualizer.py`, then open http://localhost:7860. Its sliders (RANSAC threshold, iteration count, and so on) rerun the full pipeline so you can see how each parameter affects the inliers, epipolar lines, and 3D result.

**Google Colab**

1. Upload and extract the project ZIP to `/content/`.
2. Open `hw3.ipynb` and run `%cd /content/homework_3` before the setup cell.
3. Run the cells in order. Colab provides NumPy, Matplotlib, and OpenCV with SIFT.

## Repository layout

```
two_view_reconstruction/
├── hw3.ipynb          # runs SIFT matching and the full pipeline
├── data/              # the two ZED 2 images (no precomputed matches)
├── requirements.txt
└── code/
    ├── lse.py
    ├── ransac.py
    ├── plot_epi.py
    ├── pose.py
    ├── recon3d.py
    ├── show_reproj.py
    └── visualizer.py  # optional interactive tool (provided)
```

## Tech

Python, NumPy, OpenCV (SIFT feature detection and matching only), Matplotlib.
