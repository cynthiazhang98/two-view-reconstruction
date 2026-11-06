import numpy as np
from pose import pose_candidates_from_E

def skew(t):
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])

def rot(w):
    th = np.linalg.norm(w); K = skew(w / th)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K

np.random.seed(1)
for trial in range(5):
    R_true = rot(np.random.randn(3) * 0.3)
    T_true = np.random.randn(3); T_true /= np.linalg.norm(T_true)
    E = skew(T_true) @ R_true            # convention P2 = R P1 + T
    cands = pose_candidates_from_E(E)

    print(f"--- trial {trial} ---")
    for k, c in enumerate(cands):
        det = np.linalg.det(c['R'])
        r_ok = np.allclose(c['R'], R_true, atol=1e-8)
        t_ok = np.allclose(c['T'], T_true, atol=1e-8)
        print(f"cand {k}: det(R)={det:+.3f}  R matches truth={r_ok}  T matches truth={t_ok}")