from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from bvb_predictor.data.dataset import (
    LeagueEncoders,
    fit_league_encoders,
    transform_league_categoricals,
)
from bvb_predictor.features.league_rolling import build_league_features
from bvb_predictor.models.odds_mlp import OddsProbModel


@dataclass(frozen=True)
class TrainOddsConfig:
    data_csv: str
    artifacts_path: str
    val_ratio: float
    test_ratio: float
    batch_size: int
    epochs: int
    lr: float
    patience: int
    seed: int


class LeagueOddsDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
    ) -> None:
        self._df = df.reset_index(drop=True)
        self._feature_cols = feature_cols

        x = self._df[self._feature_cols].to_numpy(dtype=np.float32)
        self._x_num = torch.tensor(x, dtype=torch.float32)

        cat = self._df[["home_team_id", "away_team_id", "league_id"]].to_numpy(
            dtype=np.int64
        )
        self._x_cat = torch.tensor(cat, dtype=torch.long)

        y = self._df[["prob_home", "prob_draw", "prob_away"]].to_numpy(dtype=np.float32)
        self._y_prob = torch.tensor(y, dtype=torch.float32)

        self._w = torch.ones(len(self._df), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._x_num[idx], self._x_cat[idx], self._y_prob[idx], self._w[idx]


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


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


def _odds_feature_cols(df: pd.DataFrame) -> list[str]:
    # Only use pre-match form/time features.
    # Do NOT include prob_* or odds_* related columns to avoid trivial learning.
    cols = [
        "month",
        "weekday",
        "season_start_year",
        "home_rest_days",
        "away_rest_days",
        "home_avg_gf_l3",
        "home_avg_ga_l3",
        "home_avg_gd_l3",
        "home_avg_gf_l5",
        "home_avg_ga_l5",
        "home_avg_gd_l5",
        "home_ema_gf",
        "home_ema_ga",
        "home_ema_gd",
        "home_avg_gf_home_l3",
        "home_avg_ga_home_l3",
        "home_avg_gd_home_l3",
        "home_ema_gf_home",
        "home_ema_ga_home",
        "home_ema_gd_home",
        "away_avg_gf_l3",
        "away_avg_ga_l3",
        "away_avg_gd_l3",
        "away_avg_gf_l5",
        "away_avg_ga_l5",
        "away_avg_gd_l5",
        "away_ema_gf",
        "away_ema_ga",
        "away_ema_gd",
        "away_avg_gf_away_l3",
        "away_avg_ga_away_l3",
        "away_avg_gd_away_l3",
        "away_ema_gf_away",
        "away_ema_ga_away",
        "away_ema_gd_away",
    ]

    seen: set[str] = set()
    out: list[str] = []
    for c in cols:
        if c in df.columns and c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _soft_ce(
    logits: torch.Tensor, target_prob: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    # Cross-entropy with soft labels.
    target = torch.clamp(target_prob, min=eps)
    target = target / torch.clamp(target.sum(dim=1, keepdim=True), min=eps)
    logp = F.log_softmax(logits, dim=1)
    return -(target * logp).sum(dim=1).mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/processed/matches.csv")
    parser.add_argument("--out", type=str, default="artifacts/odds.pt")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device: auto|cpu|cuda|cuda:0 ...",
    )

    args = parser.parse_args()
    cfg = TrainOddsConfig(
        data_csv=args.data,
        artifacts_path=args.out,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        seed=args.seed,
    )

    _set_seed(cfg.seed)

    device = _resolve_device(args.device)

    raw = pd.read_csv(Path(cfg.data_csv))
    feats = build_league_features(raw)

    # Only rows with odds-derived probs available.
    feats = feats.loc[feats["odds_available"] == 1].copy()

    if len(feats) == 0:
        raise ValueError("No rows with odds/prob available to train odds model.")

    encoders: LeagueEncoders = fit_league_encoders(feats)

    # Time split
    feats = feats.sort_values("date").reset_index(drop=True)
    train_df, val_df, test_df = _time_split(
        feats, val_ratio=cfg.val_ratio, test_ratio=cfg.test_ratio
    )

    train_df = transform_league_categoricals(train_df, encoders)
    val_df = transform_league_categoricals(val_df, encoders)
    test_df = transform_league_categoricals(test_df, encoders)

    feature_cols = _odds_feature_cols(train_df)
    if not feature_cols:
        raise ValueError("No feature columns available for odds model.")

    mu = train_df[feature_cols].mean(axis=0)
    sigma = train_df[feature_cols].std(axis=0).replace(0.0, 1.0)

    for df in (train_df, val_df, test_df):
        df[feature_cols] = (df[feature_cols] - mu) / sigma

    train_ds = LeagueOddsDataset(train_df, feature_cols=feature_cols)
    val_ds = LeagueOddsDataset(val_df, feature_cols=feature_cols)
    test_ds = LeagueOddsDataset(test_df, feature_cols=feature_cols)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    model = OddsProbModel(
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
            for x, cat, y_prob, _ in loader:
                x = x.to(device)
                cat = cat.to(device)
                y_prob = y_prob.to(device)
                logits = model(x, cat)
                losses.append(float(_soft_ce(logits, y_prob).item()))
        return float(np.mean(losses)) if losses else float("inf")

    for epoch in range(cfg.epochs):
        model.train()
        train_losses: list[float] = []

        for x, cat, y_prob, _ in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            x = x.to(device)
            cat = cat.to(device)
            y_prob = y_prob.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x, cat)
            loss = _soft_ce(logits, y_prob)
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
    test_loss = eval_loss(test_loader)

    out_path = Path(cfg.artifacts_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state_dict": model.state_dict(),
        "encoders": asdict(encoders),
        "feature_cols": feature_cols,
        "feature_mu": mu.to_dict(),
        "feature_sigma": sigma.to_dict(),
    }
    torch.save(payload, out_path)

    print(
        json.dumps({"test_loss": test_loss, "out": str(out_path)}, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
