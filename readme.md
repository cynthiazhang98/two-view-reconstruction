# CIS 5800 — HW3C, Fall 2026 (student package draft)

## 3D reconstruction from two camera views

The `data/` directory contains the **two camera images only**. Point correspondences are **not provided**: the notebook uses OpenCV SIFT detection and descriptor matching to compute them when you run the designated cells. Then implement the six functions in `code/` marked `##### STUDENT CODE START #####`.

### Run locally
1. `python -m pip install -r requirements.txt`
2. Open `hw3.ipynb` from this `homework_3/` directory and run the cells in order.
3. Optionally, run the visualizer from `code/` using `python visualizer.py`.

### Run in Google Colab
1. Upload and extract the student ZIP to `/content/`.
2. Open `hw3.ipynb` and run `%cd /content/homework_3` before the import/setup cell.
3. Run the cells in order; Colab generally provides NumPy, Matplotlib, and OpenCV with SIFT.

Submit **only** these six files from `code/`: `lse.py`, `plot_epi.py`, `pose.py`, `ransac.py`, `recon3d.py`, and `show_reproj.py`.

No VGGT, `.glb` models, precomputed matches, or wrist-camera extension are included.

Updated: the reconstruction and reprojection cells now use RANSAC inlier matches, and the notebook quick-check uses 2000 RANSAC iterations. The six Python starter files remain unfilled.
