from __future__ import annotations

import difflib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F

from bvb_predictor.features.league_rolling import build_league_features
from bvb_predictor.models.odds_mlp import OddsProbModel
from bvb_predictor.models.poisson_mlp import TeamPoissonScoreModel
from bvb_predictor.utils.score_prob import (
    nb_pmf,
    score_matrix_nb_dc,
    topk_scores,
    wdl_probs,
)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _auto_max_goals(
    lam_home: float,
    lam_away: float,
    alpha: float,
    tail_eps: float = 1e-5,
    min_cap: int = 0,
    max_cap: int = 12,
) -> int:
    def _cutoff(lam: float) -> int:
        if lam <= 0:
            return 0
        cumulative = 0.0
        k = -1
        while cumulative < (1.0 - tail_eps) and k < max_cap:
            k += 1
            cumulative += nb_pmf(k, lam, alpha)
        return k

    cut = max(_cutoff(lam_home), _cutoff(lam_away))
    cut = max(min_cap, cut)
    cut = min(max_cap, cut)
    return cut


def _collect_team_names(df: pd.DataFrame) -> list[str]:
    teams = pd.unique(pd.concat([df["home_team"], df["away_team"]], ignore_index=True))
    return sorted(t for t in teams if pd.notna(t))


def _team_suggestions(name: str, catalog: list[str], limit: int = 5) -> list[str]:
    if not catalog:
        return []
    return difflib.get_close_matches(name, catalog, n=limit, cutoff=0.6)


def _team_hint_suffix(name: str, catalog: list[str]) -> str:
    suggestions = _team_suggestions(name, catalog)
    if not suggestions:
        return ""
    return f" Closest teams: {', '.join(suggestions)}."


@dataclass(frozen=True)
class PredictRequest:
    model_path: Path
    data_path: Path
    league: str
    home_team: str
    away_team: str
    kickoff_iso: str
    device: str = "auto"
    odds_model_path: str = ""
    odds_home: float = math.nan
    odds_draw: float = math.nan
    odds_away: float = math.nan
    max_goals: int = -1
    topk: int = 10
    verbose: bool = False


def _load_main_model(
    ckpt_path: Path, device: torch.device
) -> tuple[TeamPoissonScoreModel, dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu")

    encoders = ckpt["encoders"]
    feature_cols: list[str] = ckpt["feature_cols"]
    mu = pd.Series(ckpt["feature_mu"], dtype=float)
    sigma = pd.Series(ckpt["feature_sigma"], dtype=float).replace(0.0, 1.0)

    team_to_id: dict[str, int] = encoders["team_to_id"]
    league_to_id: dict[str, int] = encoders["league_to_id"]

    model = TeamPoissonScoreModel(
        num_numerical_features=len(feature_cols),
        num_teams=len(team_to_id),
        num_leagues=len(league_to_id),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    meta = {
        "feature_cols": feature_cols,
        "feature_mu": mu,
        "feature_sigma": sigma,
        "team_to_id": team_to_id,
        "league_to_id": league_to_id,
    }
    return model, meta


def _maybe_run_odds_model(
    req: PredictRequest,
    raw: pd.DataFrame,
    upcoming_dict: dict[str, Any],
    device: torch.device,
    team_catalog: list[str],
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    need_pseudo_prob = (
        req.odds_model_path
        and (not pd.notna(req.odds_home))
        and (not pd.notna(req.odds_draw))
        and (not pd.notna(req.odds_away))
    )

    if not need_pseudo_prob:
        return raw, None

    odds_ckpt = torch.load(Path(req.odds_model_path), map_location="cpu")
    odds_feature_cols: list[str] = odds_ckpt["feature_cols"]
    odds_mu = pd.Series(odds_ckpt["feature_mu"], dtype=float)
    odds_sigma = pd.Series(odds_ckpt["feature_sigma"], dtype=float).replace(0.0, 1.0)
    odds_enc = odds_ckpt["encoders"]

    odds_team_to_id: dict[str, int] = odds_enc["team_to_id"]
    odds_league_to_id: dict[str, int] = odds_enc["league_to_id"]

    odds_model = OddsProbModel(
        num_numerical_features=len(odds_feature_cols),
        num_teams=len(odds_team_to_id),
        num_leagues=len(odds_league_to_id),
    ).to(device)
    odds_model.load_state_dict(odds_ckpt["model_state_dict"])
    odds_model.eval()

    tmp = pd.concat([raw, pd.DataFrame([upcoming_dict])], ignore_index=True)
    tmp_feats = build_league_features(tmp)
    tmp_row = tmp_feats.iloc[-1].copy()

    x_num_s = (
        tmp_row[odds_feature_cols].astype(float).fillna(odds_mu[odds_feature_cols])
    )
    x_num = (
        (x_num_s - odds_mu[odds_feature_cols]) / odds_sigma[odds_feature_cols]
    ).to_numpy()
    x_num_t = torch.tensor(x_num, dtype=torch.float32).unsqueeze(0).to(device)

    missing_odds_teams = [
        team for team in (req.home_team, req.away_team) if team not in odds_team_to_id
    ]
    if missing_odds_teams:
        hint = _team_hint_suffix(missing_odds_teams[0], team_catalog)
        names = ", ".join(missing_odds_teams)
        raise ValueError(
            f"Unseen team for odds model: {names}. Retrain odds model to include it.{hint}"
        )
    if req.league not in odds_league_to_id:
        raise ValueError(
            "Unseen league for odds model. Retrain odds model to include it."
        )

    x_cat_t = torch.tensor(
        [
            [
                odds_team_to_id[req.home_team],
                odds_team_to_id[req.away_team],
                odds_league_to_id[req.league],
            ]
        ],
        dtype=torch.long,
    ).to(device)

    with torch.no_grad():
        logits = odds_model(x_num_t, x_cat_t)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy().tolist()

    prob_home = float(probs[0])
    prob_draw = float(probs[1])
    prob_away = float(probs[2])

    upcoming_dict["prob_home"] = prob_home
    upcoming_dict["prob_draw"] = prob_draw
    upcoming_dict["prob_away"] = prob_away

    odds_model_output = {
        "probabilities": {
            "home": prob_home,
            "draw": prob_draw,
            "away": prob_away,
        },
        "decimal_odds": {
            "home": float("inf") if prob_home <= 0 else 1.0 / prob_home,
            "draw": float("inf") if prob_draw <= 0 else 1.0 / prob_draw,
            "away": float("inf") if prob_away <= 0 else 1.0 / prob_away,
        },
    }
    return tmp, odds_model_output


def predict_match(req: PredictRequest) -> dict[str, Any]:
    device = _resolve_device(req.device)
    model, meta = _load_main_model(Path(req.model_path), device)

    raw = pd.read_csv(Path(req.data_path))
    team_catalog = _collect_team_names(raw)
    raw_dates = pd.to_datetime(raw["date"], utc=True)
    upcoming_dt = pd.to_datetime(req.kickoff_iso, utc=True)
    hist = raw.loc[raw_dates < upcoming_dt].copy()

    upcoming_dict: dict[str, Any] = {
        "date": req.kickoff_iso,
        "league": req.league,
        "season": str(upcoming_dt.year % 100).zfill(2)
        + str((upcoming_dt.year + 1) % 100).zfill(2),
        "home_team": req.home_team,
        "away_team": req.away_team,
        "home_score": 0,
        "away_score": 0,
        "odds_home": req.odds_home,
        "odds_draw": req.odds_draw,
        "odds_away": req.odds_away,
    }

    hist_with_odds, odds_model_output = _maybe_run_odds_model(
        req=req,
        raw=hist,
        upcoming_dict=upcoming_dict,
        device=device,
        team_catalog=team_catalog,
    )

    feats = build_league_features(
        pd.concat([hist_with_odds, pd.DataFrame([upcoming_dict])], ignore_index=True)
    )
    row = feats.iloc[-1].copy()

    feature_cols: list[str] = meta["feature_cols"]
    mu: pd.Series = meta["feature_mu"]
    sigma: pd.Series = meta["feature_sigma"]

    x_num_s = row[feature_cols].astype(float)
    x_num = ((x_num_s - mu[feature_cols]) / sigma[feature_cols]).fillna(0.0).to_numpy()
    x_num_t = torch.tensor(x_num, dtype=torch.float32).unsqueeze(0).to(device)

    team_to_id: dict[str, int] = meta["team_to_id"]
    league_to_id: dict[str, int] = meta["league_to_id"]

    if req.home_team not in team_to_id:
        hint = _team_hint_suffix(req.home_team, team_catalog)
        raise ValueError(
            f"Unseen home team: {req.home_team}. Retrain model to include it.{hint}"
        )
    if req.away_team not in team_to_id:
        hint = _team_hint_suffix(req.away_team, team_catalog)
        raise ValueError(
            f"Unseen away team: {req.away_team}. Retrain model to include it.{hint}"
        )
    if req.league not in league_to_id:
        raise ValueError(f"Unseen league: {req.league}. Retrain model to include it.")

    x_cat_t = torch.tensor(
        [
            [
                team_to_id[req.home_team],
                team_to_id[req.away_team],
                league_to_id[req.league],
            ]
        ],
        dtype=torch.long,
    ).to(device)

    with torch.no_grad():
        mu_t, rho_t, alpha_t = model(x_num_t, x_cat_t)

    mu = mu_t.squeeze(0).cpu().numpy().tolist()
    rho = float(rho_t.squeeze(0).cpu().item())
    alpha = float(alpha_t.squeeze(0).cpu().item())

    mu_home, mu_away = float(mu[0]), float(mu[1])
    max_goals = (
        req.max_goals
        if req.max_goals >= 0
        else _auto_max_goals(mu_home, mu_away, alpha)
    )
    if req.verbose:
        print(f"[predict] max_goals={max_goals}", file=sys.stderr)

    mat = score_matrix_nb_dc(mu_home, mu_away, alpha, rho, max_goals)
    topk = topk_scores(mat, k=req.topk)
    p_home, p_draw, p_away = wdl_probs(mat)

    out: dict[str, Any] = {
        "mu_home": mu_home,
        "mu_away": mu_away,
        "rho": rho,
        "alpha": alpha,
        "max_goals": max_goals,
        "topk_scores": [{"home": h, "away": a, "p": p} for h, a, p in topk],
        "wdl": {"home_win": p_home, "draw": p_draw, "away_win": p_away},
    }

    if odds_model_output is not None:
        out["odds_model"] = odds_model_output

    return out
