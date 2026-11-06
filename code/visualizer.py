# pip install gradio opencv-python matplotlib numpy
# Run: python visualizer.py   (from Homework_3/code/)
# Then open: http://localhost:7860

# ============================================================
# Section 1 — Imports & Constants
# ============================================================
import sys
import os
import io
from contextlib import redirect_stdout

import matplotlib
matplotlib.use("Agg")  # MUST be before pyplot import — prevents backend crash in Gradio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers '3d' projection

import numpy as np
import cv2
import gradio as gr

# Ensure code directory is on the path so local modules resolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lse import least_squares_estimation
from plot_epi import plot_epipolar_lines
from recon3d import reconstruct3D
from show_reproj import show_reprojections
from pose import pose_candidates_from_E

# ── Camera intrinsics ────────────────────────────────────────
K = np.array([[967,   0, 640],
              [  0, 967, 360],
              [  0,   0,   1]], dtype=float)
K_INV = np.linalg.inv(K)

DIST_COEFFS = np.zeros(5)  # input views are rectified; match notebook assumptions

_CODE_DIR       = os.path.dirname(os.path.abspath(__file__))
IMG_LEFT_PATH   = os.path.join(_CODE_DIR, '..', 'data', 'left_30.jpg')
IMG_RIGHT_PATH  = os.path.join(_CODE_DIR, '..', 'data', 'right_30.jpg')


# ============================================================
# Section 2 — Startup Pre-computation (globals, run once)
# ============================================================
def _startup():
    """Load images, detect SIFT features, compute all BF matches once."""
    for path, label in [(IMG_LEFT_PATH, 'left'), (IMG_RIGHT_PATH, 'right')]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[Startup] {label} image not found: {os.path.abspath(path)}\n"
                f"Make sure to run: python visualizer.py  from  Homework_3/code/")

    img_left_raw  = cv2.imread(IMG_LEFT_PATH)
    img_right_raw = cv2.imread(IMG_RIGHT_PATH)

    img_left  = cv2.undistort(img_left_raw,  K, DIST_COEFFS)
    img_right = cv2.undistort(img_right_raw, K, DIST_COEFFS)

    sift = cv2.SIFT_create()
    gray1 = cv2.cvtColor(img_left,  cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    kp1, desc1 = sift.detectAndCompute(gray1, None)
    kp2, desc2 = sift.detectAndCompute(gray2, None)

    bf      = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches_sorted = sorted(matches, key=lambda m: m.distance)

    print("[Startup] " + "─" * 52)
    print(f"[Startup] Left  image : {img_left.shape}  ← {IMG_LEFT_PATH}")
    print(f"[Startup] Right image : {img_right.shape}  ← {IMG_RIGHT_PATH}")
    print(f"[Startup] SIFT keypoints : left={len(kp1)}, right={len(kp2)}")
    print(f"[Startup] BF matches (crossCheck) : {len(matches_sorted)}")
    print("[Startup] " + "─" * 52)

    return img_left, img_right, kp1, kp2, matches_sorted


_IMG_LEFT, _IMG_RIGHT, _KP1, _KP2, _ALL_MATCHES_SORTED = _startup()


# ============================================================
# Section 3 — Helpers
# ============================================================

def matches_to_points(matches):
    """DMatch list → (uncal1_Nx3, uncal2_Nx3) in homogeneous pixel coords."""
    pts1 = np.array([_KP1[m.queryIdx].pt for m in matches])
    pts2 = np.array([_KP2[m.trainIdx].pt for m in matches])
    ones = np.ones((len(matches), 1))
    return np.hstack([pts1, ones]), np.hstack([pts2, ones])


def calibrate_points(uncal_Nx3):
    """Nx3 pixel-homogeneous → Nx3 calibrated (K⁻¹ @ pts.T).T."""
    return (K_INV @ uncal_Nx3.T).T


def fig_to_numpy(fig):
    """Save matplotlib figure to BytesIO PNG, decode to uint8 RGB ndarray; closes fig."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    plt.close(fig)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def suppress_stdout():
    """Context manager that swallows stdout (used to silence recon3d prints)."""
    return redirect_stdout(io.StringIO())


def ransac_estimator_custom(X1, X2, n_iter, eps):
    """
    Exact mirror of ransac.py — eps is a runtime parameter instead of hardcoded.

    Uses the same:
    - RandomState seeding: seed = i * 10
    - Sampson distance formula (e3_skew variant)
    - Denominator guard: +1e-12 to prevent div-by-zero

    Returns: (best_E  3×3,  best_inliers  ndarray[int])
    """
    sample_size = 8
    e3_skew = np.array([[0, -1, 0],
                        [1,  0, 0],
                        [0,  0, 0]], dtype=float)

    best_E, best_inliers, best_count = None, None, -1

    for i in range(n_iter):
        perm  = np.random.RandomState(seed=i * 10).permutation(np.arange(X1.shape[0]))
        s_idx = perm[:sample_size]
        t_idx = perm[sample_size:]

        E = least_squares_estimation(X1[s_idx], X2[s_idx])

        X1t = X1[t_idx]
        X2t = X2[t_idx]

        d1 = (np.sum(X2t * (E @ X1t.T).T, axis=1)) ** 2 / (
             np.linalg.norm((e3_skew @ (E   @ X1t.T)).T, axis=1) ** 2 + 1e-12)
        d2 = (np.sum(X1t * (E.T @ X2t.T).T, axis=1)) ** 2 / (
             np.linalg.norm((e3_skew @ (E.T @ X2t.T)).T, axis=1) ** 2 + 1e-12)

        mask    = (d1 + d2) < eps
        inliers = np.concatenate([s_idx, t_idx[mask]])

        if len(inliers) > best_count:
            best_count    = len(inliers)
            best_E        = E
            best_inliers  = inliers

    return best_E, best_inliers


def reconstruct3D_with_override(cands, cal1, cal2, force_idx):
    """
    Triangulate 3-D points.

    force_idx = -1  → automatic selection via recon3d.reconstruct3D (stdout suppressed)
    force_idx = 0-3 → manually force a specific pose candidate (no auto-selection)

    Returns: (P1_Nx3, P2_Nx3, T_3, R_3x3)
    """
    if force_idx == -1:
        with suppress_stdout():
            return reconstruct3D(cands, cal1, cal2)

    c = cands[force_idx]
    R, T = c['R'], c['T']

    lambdas = np.zeros((2, cal1.shape[0]))
    for i in range(cal1.shape[0]):
        A = np.column_stack([R @ cal1[i], -cal2[i]])
        lambdas[:, i] = np.linalg.lstsq(A, -T, rcond=None)[0]

    P1 = lambdas[0].reshape(-1, 1) * cal1
    P2 = lambdas[1].reshape(-1, 1) * cal2
    return P1, P2, T, R


def compute_rmse(P1proj, P2proj, uncal1_Nx3, uncal2_Nx3):
    """
    Mean pixel reprojection error (both images combined).

    P2proj (Nx3 homogeneous) = cam2 3-D pts re-projected into cam1; compare vs uncal1.
    P1proj (Nx3 homogeneous) = cam1 3-D pts re-projected into cam2; compare vs uncal2.
    """
    p2 = P2proj[:, :2] / (P2proj[:, 2:3] + 1e-12)
    p1 = P1proj[:, :2] / (P1proj[:, 2:3] + 1e-12)
    err1 = np.sqrt(np.sum((p2 - uncal1_Nx3[:, :2]) ** 2, axis=1))
    err2 = np.sqrt(np.sum((p1 - uncal2_Nx3[:, :2]) ** 2, axis=1))
    return float(np.mean(np.concatenate([err1, err2])))


def _draw_lines_on_ax(ax, lines_3xN, h, w):
    """
    Port of plot_epi.plot_lines() onto a given Axes object.

    lines_3xN : 3×N array where each column is [a, b, c] (line ax+by+c=0)
    """
    for i in range(lines_3xN.shape[1]):
        a, b, c = lines_3xN[0, i], lines_3xN[1, i], lines_3xN[2, i]
        if abs(b) < 1e-10:                      # nearly vertical
            if abs(a) > 1e-10:
                x0 = -c / a
                ax.plot([x0, x0], [0, h], linewidth=0.8)
        elif abs(a / b) < 1:                     # more horizontal → clip to x-extent
            y0 = -c / b
            yw = y0 - w * a / b
            ax.plot([0, w], [y0, yw], linewidth=0.8)
        else:                                     # more vertical → clip to y-extent
            if abs(a) > 1e-10:
                x0 = -c / a
                xh = x0 - h * b / a
                ax.plot([x0, xh], [0, h], linewidth=0.8)


def _make_error_img(msg):
    """Create a single light-grey error image with red text."""
    img = np.ones((200, 640, 3), dtype=np.uint8) * 235
    y = 40
    for chunk in [msg[i:i+72] for i in range(0, min(len(msg), 288), 72)]:
        cv2.putText(img, chunk, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (160, 0, 0), 1, cv2.LINE_AA)
        y += 28
    return img


def _error_panels(msg):
    """Return (match_img, epi_img, scatter_img, reproj_img, stats_str) for error states."""
    img = _make_error_img(msg)
    return img, img.copy(), img.copy(), img.copy(), f"ERROR\n{'─'*44}\n{msg}"


# ============================================================
# Section 4 — Render Helpers  (all return H×W×3 uint8 RGB)
# ============================================================

def _render_match_panel(matches):
    """Draw feature matches with an inlier-count badge."""
    n_disp = min(len(matches), 100)
    out = cv2.drawMatches(
        _IMG_LEFT,  _KP1,
        _IMG_RIGHT, _KP2,
        matches[:n_disp], None,
        matchColor=(0, 220, 0),
        singlePointColor=(200, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.putText(out, f"Matches: {len(matches)}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 220, 220), 2, cv2.LINE_AA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def _render_epipolar_panel(uncal1_3xN, uncal2_3xN, E, num_lines):
    """
    Compute epipolar lines (via plot_epi.plot_epipolar_lines with plot=False),
    then draw them on a 1×2 figure.

    uncal1_3xN / uncal2_3xN : 3×N arrays of selected inlier pixel-homogeneous points
    """
    lines1, lines2 = plot_epipolar_lines(
        _IMG_LEFT, _IMG_RIGHT, uncal1_3xN, uncal2_3xN, E, K, plot=False)

    n  = min(num_lines, lines1.shape[1])
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax, img, lines, title in [
        (axes[0], _IMG_LEFT,  lines1[:, :n], "Epipolar Lines – Left"),
        (axes[1], _IMG_RIGHT, lines2[:, :n], "Epipolar Lines – Right"),
    ]:
        ax.imshow(img[:, :, ::-1])
        ax.set_xlim([0, img.shape[1]])
        ax.set_ylim([img.shape[0], 0])
        ax.set_title(title, fontsize=10)
        ax.axis('off')
        _draw_lines_on_ax(ax, lines, img.shape[0], img.shape[1])

    plt.tight_layout()
    return fig_to_numpy(fig)


def _render_scatter_panel(P1, P2, R, T):
    """3-D scatter of triangulated points + camera origin markers."""
    fig = plt.figure(figsize=(7, 6))
    ax  = fig.add_subplot(111, projection='3d')

    def _clean(P):
        return P[np.all(np.isfinite(P), axis=1)]

    P1c = _clean(P1)
    P2c = _clean(P2)

    if P1c.shape[0]:
        ax.scatter(P1c[:, 0], P1c[:, 1], P1c[:, 2],
                   c='steelblue', alpha=0.4, s=8, label=f'Cam1 pts ({P1c.shape[0]})')
    if P2c.shape[0]:
        ax.scatter(P2c[:, 0], P2c[:, 1], P2c[:, 2],
                   c='tomato', alpha=0.4, s=8, label=f'Cam2 pts ({P2c.shape[0]})')

    # Camera origins
    ax.scatter([0], [0], [0], c='lime',   s=140, marker='^', zorder=5, label='Cam1 origin')
    cam2_world = -R.T @ T
    ax.scatter([cam2_world[0]], [cam2_world[1]], [cam2_world[2]],
               c='orange', s=140, marker='^', zorder=5, label='Cam2 origin')

    ax.set_title("3D Reconstruction", fontsize=10)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.legend(fontsize=7, loc='upper left')
    plt.tight_layout()
    return fig_to_numpy(fig)


def _render_reproj_panel(P1proj, P2proj, uncal1_3xN, uncal2_3xN):
    """
    1x2 figure: blue squares = reprojected points, red circles = original detections.

    P1proj (Nx3) = cam1 3-D pts projected to cam2 image plane → left vs uncal2
    P2proj (Nx3) = cam2 3-D pts projected to cam1 image plane → right vs uncal1
    """
    p2_px = P2proj[:, :2] / (P2proj[:, 2:3] + 1e-12)   # cam2 → cam1 image
    p1_px = P1proj[:, :2] / (P1proj[:, 2:3] + 1e-12)   # cam1 → cam2 image

    h, w  = _IMG_LEFT.shape[:2]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax, img, reproj_px, orig_3xN, title in [
        (axes[0], _IMG_LEFT,  p2_px, uncal1_3xN, "Reprojection – Left (cam1)"),
        (axes[1], _IMG_RIGHT, p1_px, uncal2_3xN, "Reprojection – Right (cam2)"),
    ]:
        ax.imshow(img[:, :, ::-1])
        ax.set_xlim([0, w])
        ax.set_ylim([h, 0])
        ax.set_title(title, fontsize=10)
        ax.axis('off')
        ax.plot(reproj_px[:, 0], reproj_px[:, 1],
                'bs', markersize=3, alpha=0.7, label='Reprojected')
        ax.plot(orig_3xN[0, :], orig_3xN[1, :],
                'ro', markersize=3, alpha=0.7, label='Original')
        ax.legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    return fig_to_numpy(fig)


# ============================================================
# Section 5 — run_pipeline
# ============================================================

def _format_stats(E_lse, E_ransac, n_matches, n_inliers, rmse, R, T):
    sep = "─" * 46
    lines = [
        sep,
        f"  Matches used    : {n_matches}",
        f"  RANSAC inliers  : {n_inliers}",
        f"  RMSE (pixels)   : {'NaN' if np.isnan(rmse) else f'{rmse:.3f}'}",
        f"  det(R)          : {np.linalg.det(R):+.6f}",
        f"  T               : [{T[0]:+.4f}  {T[1]:+.4f}  {T[2]:+.4f}]",
        sep,
        "  E (LSE):",
    ]
    for row in E_lse:
        lines.append(f"    [{row[0]:+.4f}  {row[1]:+.4f}  {row[2]:+.4f}]")
    lines.append("  E (RANSAC):")
    for row in E_ransac:
        lines.append(f"    [{row[0]:+.4f}  {row[1]:+.4f}  {row[2]:+.4f}]")
    lines.append(sep)
    return "\n".join(lines)


def run_pipeline(max_matches, eps_log, num_iterations, num_lines, line_sample, force_candidate):
    """
    Master pipeline function wired to every Gradio control.

    Inputs
    ------
    max_matches     : int   — top-N BF matches to use
    eps_log         : float — log10(Sampson threshold) for RANSAC
    num_iterations  : int   — RANSAC iteration count
    num_lines       : int   — epipolar lines to draw
    line_sample     : str   — "first N" | "random N"
    force_candidate : str   — "Auto" | "Candidate 0..3"

    Outputs  (5-tuple)
    -------
    match_img, epi_img, scatter_img, reproj_img : HxWx3 uint8 RGB ndarrays
    stats_str                                   : formatted statistics string
    """

    # ── Stage 1: Match selection ──────────────────────────────────
    n       = min(int(max_matches), len(_ALL_MATCHES_SORTED))
    matches = _ALL_MATCHES_SORTED[:n]

    if len(matches) < 8:
        return _error_panels("Need ≥8 matches. Increase max_matches.")

    uncal1_Nx3, uncal2_Nx3 = matches_to_points(matches)
    cal1 = calibrate_points(uncal1_Nx3)
    cal2 = calibrate_points(uncal2_Nx3)

    # ── Stage 2: LSE (stored for stats display only) ──────────────
    E_lse = least_squares_estimation(cal1, cal2)

    # ── Stage 3: RANSAC ───────────────────────────────────────────
    eps = 10.0 ** eps_log
    try:
        E_ransac, inlier_idx = ransac_estimator_custom(
            cal1, cal2, int(num_iterations), eps)
    except Exception as exc:
        return _error_panels(f"RANSAC failed: {exc}")

    if inlier_idx is None or len(inlier_idx) < 8:
        return _error_panels(
            "Too few inliers. Try increasing eps (move slider right) or num_iterations.")

    # ── Stage 4: Inlier subsets ───────────────────────────────────
    inlier_uncal1 = uncal1_Nx3[inlier_idx]   # Nx3
    inlier_uncal2 = uncal2_Nx3[inlier_idx]   # Nx3
    inlier_cal1   = cal1[inlier_idx]          # Nx3
    inlier_cal2   = cal2[inlier_idx]          # Nx3
    n_inliers     = len(inlier_idx)

    # ── Stage 5: Line selection for epipolar display ──────────────
    n_lines = min(int(num_lines), n_inliers)
    if line_sample == "random N":
        rng      = np.random.RandomState(seed=42)
        line_idx = rng.choice(n_inliers, n_lines, replace=False)
    else:   # "first N"
        line_idx = np.arange(n_lines)

    # ── Stage 6: Match panel ──────────────────────────────────────
    match_img = _render_match_panel(matches)

    # ── Stage 7: Epipolar panel ───────────────────────────────────
    try:
        epi_img = _render_epipolar_panel(
            inlier_uncal1[line_idx].T,   # 3×n_lines
            inlier_uncal2[line_idx].T,   # 3×n_lines
            E_ransac, n_lines)
    except Exception as exc:
        epi_img = _make_error_img(f"Epipolar render failed: {exc}")

    # ── Stage 8: Pose + 3-D reconstruction ───────────────────────
    force_idx = (-1 if force_candidate == "Auto"
                 else int(force_candidate.split()[-1]))
    cands = pose_candidates_from_E(E_ransac)

    R, T_vec    = np.eye(3), np.zeros(3)
    scatter_img = _make_error_img("Reconstruction pending…")
    reproj_img  = _make_error_img("Reconstruction pending…")
    rmse        = float('nan')

    try:
        P1, P2, T_vec, R = reconstruct3D_with_override(
            cands, inlier_cal1, inlier_cal2, force_idx)
        scatter_img = _render_scatter_panel(P1, P2, R, T_vec)

        # show_reprojections expects uncalibrated as 3×N
        P1proj, P2proj = show_reprojections(
            _IMG_LEFT, _IMG_RIGHT,
            inlier_uncal1.T, inlier_uncal2.T,   # 3×N
            P1, P2, K, T_vec, R, plot=False)

        rmse       = compute_rmse(P1proj, P2proj, inlier_uncal1, inlier_uncal2)
        reproj_img = _render_reproj_panel(
            P1proj, P2proj,
            inlier_uncal1.T, inlier_uncal2.T)   # 3×N for axis indexing

    except Exception as exc:
        scatter_img = _make_error_img(f"Reconstruction failed: {exc}")
        reproj_img  = _make_error_img(f"Reprojection failed: {exc}")

    stats = _format_stats(E_lse, E_ransac, n, n_inliers, rmse, R, T_vec)
    return match_img, epi_img, scatter_img, reproj_img, stats


# ============================================================
# Section 6 — Gradio UI  (gr.Blocks)
# ============================================================

with gr.Blocks(title="CIS 5800 HW3 — 3D Reconstruction Visualizer") as demo:

    gr.Markdown("## CIS 5800 HW3 — Interactive 3D Reconstruction Visualizer")
    gr.Markdown(
        "Adjust any control to rerun the full pipeline in real-time.  "
        "**Left column** = algorithm parameters · **Right** = visualizations")

    with gr.Row():

        # ── Left column: controls ─────────────────────────────────
        with gr.Column(scale=1):

            with gr.Accordion("LSE — Feature Matching", open=True):
                max_matches = gr.Slider(
                    minimum=20, maximum=2000, value=200, step=1,
                    label="max_matches  (top-N BF matches used for LSE & RANSAC)")

            with gr.Accordion("RANSAC", open=True):
                eps_log = gr.Slider(
                    minimum=-7, maximum=-3, value=-5, step=0.1,
                    label="log₁₀(ε)  [Sampson distance threshold]")
                num_iterations = gr.Slider(
                    minimum=50, maximum=60000, value=60000, step=50,
                    label="num_iterations")

            with gr.Accordion("Epipolar Lines", open=True):
                num_lines = gr.Slider(
                    minimum=3, maximum=500, value=20, step=1,
                    label="num_lines  (inlier lines to draw)")
                line_sample = gr.Dropdown(
                    choices=["first N", "random N"],
                    value="first N",
                    label="line_sample")

            with gr.Accordion("Pose / 3D Reconstruction", open=True):
                force_candidate = gr.Radio(
                    choices=["Auto",
                             "Candidate 0", "Candidate 1",
                             "Candidate 2", "Candidate 3"],
                    value="Auto",
                    label="force_candidate  (override auto pose selector)")

        # ── Right column: image outputs ───────────────────────────
        with gr.Column(scale=3):
            with gr.Row():
                match_out   = gr.Image(label="Feature Matches")
                epi_out     = gr.Image(label="Epipolar Lines")
            with gr.Row():
                scatter_out = gr.Image(label="3D Scatter")
                reproj_out  = gr.Image(label="Reprojections")

    # ── Full-width stats ──────────────────────────────────────────
    stats_out = gr.Textbox(label="Statistics", lines=12)

    # ── Wire controls ─────────────────────────────────────────────
    _inputs  = [max_matches, eps_log, num_iterations,
                num_lines, line_sample, force_candidate]
    _outputs = [match_out, epi_out, scatter_out, reproj_out, stats_out]

    for ctrl in _inputs:
        ctrl.change(fn=run_pipeline, inputs=_inputs, outputs=_outputs)

    demo.load(fn=run_pipeline, inputs=_inputs, outputs=_outputs)


if __name__ == "__main__":
    demo.launch(server_name="localhost", server_port=7860, share=False)
