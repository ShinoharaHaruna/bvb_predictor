from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from bvb_predictor.data.dataset import (
    LeagueScoreDataset,
    fit_league_encoders,
    transform_league_categoricals,
)
from bvb_predictor.features.league_rolling import build_league_features
from bvb_predictor.models.poisson_mlp import (
    TeamPoissonScoreModel,
    poisson_nll,
    poisson_nll_weighted,
)
from bvb_predictor.utils.score_prob import score_matrix, topk_scores, wdl_probs


@dataclass(frozen=True)
class TrainConfig:
    data_csv: str
    artifacts_dir: str
    val_ratio: float
    test_ratio: float
    batch_size: int
    epochs: int
    lr: float
    patience: int
    seed: int
    max_goals: int
    topk: int
    time_decay_tau_days: float
    finetune_recent_seasons: int
    finetune_epochs: int
    finetune_lr: float


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _time_split(
    df: pd.DataFrame, val_ratio: float, test_ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))

    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError("Not enough data for time split.")

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train : n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val :].copy()

    return train_df, val_df, test_df


def _feature_cols(df: pd.DataFrame) -> list[str]:
    cols: list[str] = [
        "month",
        "weekday",
        "season_start_year",
        "home_rest_days",
        "away_rest_days",
        "home_avg_gf_l5",
        "home_avg_ga_l5",
        "home_avg_gf_l10",
        "home_avg_ga_l10",
        "home_avg_gf_home_l10",
        "home_avg_ga_home_l10",
        "home_avg_gd_l5",
        "home_avg_gd_l10",
        "home_avg_gd_home_l10",
        "away_avg_gf_l5",
        "away_avg_ga_l5",
        "away_avg_gf_l10",
        "away_avg_ga_l10",
        "away_avg_gf_away_l10",
        "away_avg_ga_away_l10",
        "away_avg_gd_l5",
        "away_avg_gd_l10",
        "away_avg_gd_away_l10",
        "prob_home",
        "prob_draw",
        "prob_away",
        "odds_overround",
        "odds_available",
    ]

    # Keep stable order
    seen: set[str] = set()
    ordered: list[str] = []
    for c in cols:
        if c in df.columns and c not in seen:
            ordered.append(c)
            seen.add(c)
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/processed/matches.csv")
    parser.add_argument("--artifacts", type=str, default="artifacts")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-goals", type=int, default=6)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument(
        "--time-decay-tau-days",
        type=float,
        default=0.0,
        help="If >0, apply exp(-days_ago/tau) sample weighting on training loss.",
    )
    parser.add_argument(
        "--finetune-recent-seasons",
        type=int,
        default=0,
        help="If >0, finetune on the last N seasons within the train split.",
    )
    parser.add_argument("--finetune-epochs", type=int, default=30)
    parser.add_argument("--finetune-lr", type=float, default=3e-4)

    args = parser.parse_args()

    cfg = TrainConfig(
        data_csv=args.data,
        artifacts_dir=args.artifacts,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        seed=args.seed,
        max_goals=args.max_goals,
        topk=args.topk,
        time_decay_tau_days=args.time_decay_tau_days,
        finetune_recent_seasons=args.finetune_recent_seasons,
        finetune_epochs=args.finetune_epochs,
        finetune_lr=args.finetune_lr,
    )

    _set_seed(cfg.seed)

    data_path = Path(cfg.data_csv)
    if not data_path.exists():
        raise FileNotFoundError(
            f"{cfg.data_csv} not found. Please create it following README schema."
        )

    raw = pd.read_csv(data_path)
    feats = build_league_features(raw)

    # Team/league ID mapping is metadata, fitting on full dataset does not leak targets.
    # This avoids unseen team IDs in validation/test time splits.
    encoders = fit_league_encoders(feats)

    train_df, val_df, test_df = _time_split(
        feats, val_ratio=cfg.val_ratio, test_ratio=cfg.test_ratio
    )

    train_df = transform_league_categoricals(train_df, encoders)
    val_df = transform_league_categoricals(val_df, encoders)
    test_df = transform_league_categoricals(test_df, encoders)

    # Time-decay sample weight on train only.
    if cfg.time_decay_tau_days and cfg.time_decay_tau_days > 0:
        train_dates = pd.to_datetime(train_df["date"], utc=True)
        ref = train_dates.max()
        days_ago = (ref - train_dates).dt.total_seconds() / 86400.0
        train_df["sample_weight"] = np.exp(
            -days_ago.to_numpy(dtype=np.float64) / cfg.time_decay_tau_days
        ).astype(np.float32)

    feature_cols = _feature_cols(train_df)

    # Standardize numerical features by train set only.
    mu = train_df[feature_cols].mean(axis=0)
    sigma = train_df[feature_cols].std(axis=0).replace(0.0, 1.0)

    for df in (train_df, val_df, test_df):
        df[feature_cols] = (df[feature_cols] - mu) / sigma

    train_ds = LeagueScoreDataset(train_df, feature_cols=feature_cols)
    val_ds = LeagueScoreDataset(val_df, feature_cols=feature_cols)
    test_ds = LeagueScoreDataset(test_df, feature_cols=feature_cols)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TeamPoissonScoreModel(
        num_numerical_features=len(feature_cols),
        num_teams=len(encoders.team_to_id),
        num_leagues=len(encoders.league_to_id),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0

    def eval_loss(loader: DataLoader) -> float:
        model.eval()
        losses: list[float] = []
        with torch.no_grad():
            for x, cat, y, _ in loader:
                x = x.to(device)
                cat = cat.to(device)
                y = y.to(device)
                lam = model(x, cat)
                losses.append(float(poisson_nll(y, lam).item()))
        return float(np.mean(losses)) if losses else float("inf")

    def eval_metrics(loader: DataLoader) -> dict[str, float]:
        model.eval()

        mae_home: list[float] = []
        mae_away: list[float] = []

        wdl_correct = 0
        exact_top1 = 0
        exact_topk = 0
        total = 0

        with torch.no_grad():
            for x, cat, y, _ in loader:
                x = x.to(device)
                cat = cat.to(device)
                y = y.to(device)

                lam = model(x, cat)
                y_cpu = y.detach().cpu().numpy()
                lam_cpu = lam.detach().cpu().numpy()

                # MAE on lambda
                mae_home.extend(np.abs(lam_cpu[:, 0] - y_cpu[:, 0]).tolist())
                mae_away.extend(np.abs(lam_cpu[:, 1] - y_cpu[:, 1]).tolist())

                for i in range(len(y_cpu)):
                    yh = int(round(float(y_cpu[i, 0])))
                    ya = int(round(float(y_cpu[i, 1])))

                    mat = score_matrix(
                        float(lam_cpu[i, 0]), float(lam_cpu[i, 1]), cfg.max_goals
                    )
                    top = topk_scores(mat, k=cfg.topk)
                    if top:
                        h0, a0, _ = top[0]
                        if h0 == yh and a0 == ya:
                            exact_top1 += 1

                        if any((h == yh and a == ya) for h, a, _ in top):
                            exact_topk += 1

                    p_home, p_draw, p_away = wdl_probs(mat)
                    pred = int(np.argmax([p_home, p_draw, p_away]))
                    true = 0 if yh > ya else 1 if yh == ya else 2
                    if pred == true:
                        wdl_correct += 1

                    total += 1

        if total == 0:
            return {}

        return {
            "mae_home": float(np.mean(mae_home)) if mae_home else float("nan"),
            "mae_away": float(np.mean(mae_away)) if mae_away else float("nan"),
            "wdl_accuracy": float(wdl_correct / total),
            "exact_score_top1_accuracy": float(exact_top1 / total),
            f"exact_score_top{cfg.topk}_accuracy": float(exact_topk / total),
        }

    for epoch in range(cfg.epochs):
        model.train()
        train_losses: list[float] = []

        for x, cat, y, w in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            x = x.to(device)
            cat = cat.to(device)
            y = y.to(device)
            w = w.to(device)

            optimizer.zero_grad(set_to_none=True)
            lam = model(x, cat)
            if cfg.time_decay_tau_days and cfg.time_decay_tau_days > 0:
                loss = poisson_nll_weighted(y, lam, w)
            else:
                loss = poisson_nll(y, lam)
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))

        val_loss = eval_loss(val_loader)
        train_loss = float(np.mean(train_losses)) if train_losses else float("inf")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val": best_val,
                    "bad_epochs": bad_epochs,
                },
                ensure_ascii=False,
            )
        )

        if bad_epochs >= cfg.patience:
            break

    if best_state is None:
        raise RuntimeError("Training failed: no best checkpoint.")

    model.load_state_dict(best_state)

    # Optional finetune on recent seasons within train split
    if (
        cfg.finetune_recent_seasons
        and cfg.finetune_recent_seasons > 0
        and "season" in train_df.columns
    ):
        seasons = sorted(train_df["season"].astype(str).unique().tolist())
        recent = set(seasons[-cfg.finetune_recent_seasons :])
        ft_df = train_df.loc[train_df["season"].astype(str).isin(recent)].copy()
        if len(ft_df) > 0:
            ft_ds = LeagueScoreDataset(ft_df, feature_cols=feature_cols)
            ft_loader = DataLoader(ft_ds, batch_size=cfg.batch_size, shuffle=False)

            ft_optim = torch.optim.Adam(model.parameters(), lr=cfg.finetune_lr)
            for _ in range(cfg.finetune_epochs):
                model.train()
                for x, cat, y, w in ft_loader:
                    x = x.to(device)
                    cat = cat.to(device)
                    y = y.to(device)
                    w = w.to(device)
                    ft_optim.zero_grad(set_to_none=True)
                    lam = model(x, cat)
                    if cfg.time_decay_tau_days and cfg.time_decay_tau_days > 0:
                        loss = poisson_nll_weighted(y, lam, w)
                    else:
                        loss = poisson_nll(y, lam)
                    loss.backward()
                    ft_optim.step()
    test_loss = eval_loss(test_loader)
    val_metrics = eval_metrics(val_loader)
    test_metrics = eval_metrics(test_loader)

    print(
        json.dumps(
            {"test_loss": test_loss, "val": val_metrics, "test": test_metrics},
            ensure_ascii=False,
        )
    )

    artifacts_dir = Path(cfg.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state_dict": model.state_dict(),
        "encoders": asdict(encoders),
        "feature_cols": feature_cols,
        "feature_mu": mu.to_dict(),
        "feature_sigma": sigma.to_dict(),
    }

    torch.save(payload, artifacts_dir / "model.pt")

    metrics = {
        "best_val_loss": best_val,
        "test_loss": test_loss,
        "val": val_metrics,
        "test": test_metrics,
        "max_goals": cfg.max_goals,
        "topk": cfg.topk,
    }
    (artifacts_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
