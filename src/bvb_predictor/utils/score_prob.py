from __future__ import annotations

import math

import numpy as np


def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 0.0
    return (lam**k) * math.exp(-lam) / math.factorial(k)


def score_matrix(lam_home: float, lam_away: float, max_goals: int) -> np.ndarray:
    p_home = np.array(
        [poisson_pmf(k, lam_home) for k in range(max_goals + 1)], dtype=np.float64
    )
    p_away = np.array(
        [poisson_pmf(k, lam_away) for k in range(max_goals + 1)], dtype=np.float64
    )
    mat = np.outer(p_home, p_away)

    s = float(mat.sum())
    if s > 0:
        mat = mat / s

    return mat


def topk_scores(mat: np.ndarray, k: int) -> list[tuple[int, int, float]]:
    flat = mat.ravel()
    if k >= flat.size:
        idx = np.argsort(-flat)
    else:
        idx = np.argpartition(-flat, k)[:k]
        idx = idx[np.argsort(-flat[idx])]

    rows, cols = np.unravel_index(idx, mat.shape)
    return [
        (int(r), int(c), float(mat[r, c])) for r, c in zip(rows, cols, strict=False)
    ]


def wdl_probs(mat: np.ndarray) -> tuple[float, float, float]:
    # home_win, draw, away_win
    home_win = float(np.tril(mat, k=-1).sum())
    draw = float(np.trace(mat))
    away_win = float(np.triu(mat, k=1).sum())
    return home_win, draw, away_win
