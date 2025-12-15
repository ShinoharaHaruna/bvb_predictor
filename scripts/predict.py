from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from bvb_predictor.features.league_rolling import build_league_features
from bvb_predictor.models.poisson_mlp import TeamPoissonScoreModel
from bvb_predictor.models.odds_mlp import OddsProbModel
from bvb_predictor.utils.score_prob import score_matrix_nb_dc, topk_scores, wdl_probs


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="artifacts/model.pt")
    parser.add_argument("--data", type=str, default="data/processed/matches.csv")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device: auto|cpu|cuda|cuda:0 ...",
    )
    parser.add_argument("--league", type=str, required=True)
    parser.add_argument("--home", type=str, required=True)
    parser.add_argument("--away", type=str, required=True)
    parser.add_argument("--date", type=str, required=True)
    parser.add_argument("--odds-home", type=float, default=float("nan"))
    parser.add_argument("--odds-draw", type=float, default=float("nan"))
    parser.add_argument("--odds-away", type=float, default=float("nan"))
    parser.add_argument(
        "--odds-model",
        type=str,
        default="",
        help="Optional odds model checkpoint. If set and odds are omitted, use it to predict prob_home/draw/away.",
    )
    parser.add_argument("--max-goals", type=int, default=6)
    parser.add_argument("--topk", type=int, default=10)

    args = parser.parse_args()

    device = _resolve_device(args.device)

    ckpt = torch.load(Path(args.model), map_location="cpu")

    encoders = ckpt["encoders"]
    feature_cols: list[str] = ckpt["feature_cols"]
    mu = pd.Series(ckpt["feature_mu"], dtype=float)
    sigma = pd.Series(ckpt["feature_sigma"], dtype=float).replace(0.0, 1.0)

    team_to_id: dict[str, int] = encoders["team_to_id"]
    league_to_id: dict[str, int] = encoders["league_to_id"]

    raw = pd.read_csv(Path(args.data))

    raw_dates = pd.to_datetime(raw["date"], utc=True)
    upcoming_dt = pd.to_datetime(args.date, utc=True)
    raw = raw.loc[raw_dates < upcoming_dt].copy()

    upcoming_dict: dict[str, object] = {
        "date": args.date,
        "league": args.league,
        "season": str(pd.to_datetime(args.date, utc=True).year % 100).zfill(2)
        + str((pd.to_datetime(args.date, utc=True).year + 1) % 100).zfill(2),
        "home_team": args.home,
        "away_team": args.away,
        "home_score": 0,
        "away_score": 0,
        "odds_home": args.odds_home,
        "odds_draw": args.odds_draw,
        "odds_away": args.odds_away,
    }

    need_pseudo_prob = (
        args.odds_model
        and (not pd.notna(args.odds_home))
        and (not pd.notna(args.odds_draw))
        and (not pd.notna(args.odds_away))
    )

    if need_pseudo_prob:
        odds_ckpt = torch.load(Path(args.odds_model), map_location="cpu")
        odds_feature_cols: list[str] = odds_ckpt["feature_cols"]
        odds_mu = pd.Series(odds_ckpt["feature_mu"], dtype=float)
        odds_sigma = pd.Series(odds_ckpt["feature_sigma"], dtype=float).replace(
            0.0, 1.0
        )
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

        if args.home not in odds_team_to_id or args.away not in odds_team_to_id:
            raise ValueError(
                "Unseen team for odds model. Retrain odds model to include it."
            )
        if args.league not in odds_league_to_id:
            raise ValueError(
                "Unseen league for odds model. Retrain odds model to include it."
            )

        x_cat_t = torch.tensor(
            [
                [
                    odds_team_to_id[args.home],
                    odds_team_to_id[args.away],
                    odds_league_to_id[args.league],
                ]
            ],
            dtype=torch.long,
        ).to(device)

        with torch.no_grad():
            logits = odds_model(x_num_t, x_cat_t)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy().tolist()

        upcoming_dict["prob_home"] = float(probs[0])
        upcoming_dict["prob_draw"] = float(probs[1])
        upcoming_dict["prob_away"] = float(probs[2])

    feats = build_league_features(
        pd.concat([raw, pd.DataFrame([upcoming_dict])], ignore_index=True)
    )
    row = feats.iloc[-1].copy()

    if args.home not in team_to_id:
        raise ValueError(f"Unseen team: {args.home}. Retrain model to include it.")
    if args.away not in team_to_id:
        raise ValueError(f"Unseen team: {args.away}. Retrain model to include it.")
    if args.league not in league_to_id:
        raise ValueError(f"Unseen league: {args.league}. Retrain model to include it.")

    x_num_s = row[feature_cols].astype(float)
    # Safety: if some feature is still NaN, impute with train mean.
    x_num_s = x_num_s.fillna(mu[feature_cols])
    x_num = ((x_num_s - mu[feature_cols]) / sigma[feature_cols]).to_numpy()
    x_num_t = torch.tensor(x_num, dtype=torch.float32).unsqueeze(0).to(device)

    home_id = team_to_id[args.home]
    away_id = team_to_id[args.away]
    league_id = league_to_id[args.league]
    x_cat_t = torch.tensor([[home_id, away_id, league_id]], dtype=torch.long).to(device)

    model = TeamPoissonScoreModel(
        num_numerical_features=len(feature_cols),
        num_teams=len(team_to_id),
        num_leagues=len(league_to_id),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        mu_t, rho_t, alpha_t = model(x_num_t, x_cat_t)

    mu = mu_t.squeeze(0).cpu().numpy().tolist()
    rho = float(rho_t.squeeze(0).cpu().item())
    alpha = float(alpha_t.squeeze(0).cpu().item())

    mu_home, mu_away = float(mu[0]), float(mu[1])
    mat = score_matrix_nb_dc(
        mu_home, mu_away, alpha=alpha, rho=rho, max_goals=args.max_goals
    )
    topk = topk_scores(mat, k=args.topk)
    p_home, p_draw, p_away = wdl_probs(mat)

    out = {
        "mu_home": mu_home,
        "mu_away": mu_away,
        "rho": rho,
        "alpha": alpha,
        "topk_scores": [{"home": h, "away": a, "p": p} for h, a, p in topk],
        "wdl": {"home_win": p_home, "draw": p_draw, "away_win": p_away},
    }

    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
