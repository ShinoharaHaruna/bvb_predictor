from __future__ import annotations

import math

import numpy as np


def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 0.0
    return (lam**k) * math.exp(-lam) / math.factorial(k)


def nb_pmf(k: int, mu: float, alpha: float) -> float:
    if mu <= 0:
        return 0.0
    if alpha <= 0:
        return poisson_pmf(k, mu)

    r = 1.0 / alpha
    p = r / (r + mu)
    # log pmf to avoid overflow
    logp = (
        math.lgamma(k + r)
        - math.lgamma(r)
        - math.lgamma(k + 1)
        + r * math.log(p)
        + k * math.log(max(1.0 - p, 1e-12))
    )
    return float(math.exp(logp))


def _dc_tau(x: int, y: int, lam_home: float, lam_away: float, rho: float) -> float:
    if x == 0 and y == 0:
        return 1.0 - (lam_home * lam_away * rho)
    if x == 0 and y == 1:
        return 1.0 + (lam_home * rho)
    if x == 1 and y == 0:
        return 1.0 + (lam_away * rho)
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


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


def score_matrix_nb_dc(
    lam_home: float, lam_away: float, alpha: float, rho: float, max_goals: int
) -> np.ndarray:
    p_home = np.array(
        [nb_pmf(k, lam_home, alpha) for k in range(max_goals + 1)], dtype=np.float64
    )
    p_away = np.array(
        [nb_pmf(k, lam_away, alpha) for k in range(max_goals + 1)], dtype=np.float64
    )

    mat = np.outer(p_home, p_away)
    for x in range(min(2, max_goals + 1)):
        for y in range(min(2, max_goals + 1)):
            mat[x, y] *= _dc_tau(x, y, lam_home, lam_away, rho)

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
