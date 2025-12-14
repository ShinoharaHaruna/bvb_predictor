from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LeagueRollingConfig:
    l5: int = 5
    l10: int = 10


def build_league_features(
    matches: pd.DataFrame, cfg: LeagueRollingConfig | None = None
) -> pd.DataFrame:
    if cfg is None:
        cfg = LeagueRollingConfig()

    df = matches.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, format="mixed")
    df = df.sort_values("date").reset_index(drop=True)

    # Base columns
    df["month"] = df["date"].dt.month.astype(np.int64)
    df["weekday"] = df["date"].dt.weekday.astype(np.int64)

    # Season as a coarse time feature (football-data season code like "2425").
    if "season" in df.columns:
        s = df["season"].astype(str)
        # YY?? -> 2000+YY for modern seasons, otherwise 1900+YY.
        yy = pd.to_numeric(s.str.slice(0, 2), errors="coerce")
        season_start = np.where(yy >= 70, 1900 + yy, 2000 + yy)
        df["season_start_year"] = season_start.astype(np.int64)
    else:
        df["season_start_year"] = df["date"].dt.year.astype(np.int64)

    # Odds
    for c in ("odds_home", "odds_draw", "odds_away"):
        if c not in df.columns:
            df[c] = np.nan
    df["odds_available"] = (
        df[["odds_home", "odds_draw", "odds_away"]].notna().all(axis=1).astype(np.int64)
    )

    inv_home = 1.0 / df["odds_home"].astype(np.float64)
    inv_draw = 1.0 / df["odds_draw"].astype(np.float64)
    inv_away = 1.0 / df["odds_away"].astype(np.float64)
    inv_sum = inv_home + inv_draw + inv_away

    df["odds_overround"] = inv_sum.astype(np.float64)
    df["prob_home"] = (inv_home / inv_sum).astype(np.float64)
    df["prob_draw"] = (inv_draw / inv_sum).astype(np.float64)
    df["prob_away"] = (inv_away / inv_sum).astype(np.float64)

    # Long format for team rolling
    home_rows = pd.DataFrame(
        {
            "match_idx": df.index,
            "date": df["date"],
            "team": df["home_team"],
            "is_home": 1,
            "gf": df["home_score"].astype(np.float32),
            "ga": df["away_score"].astype(np.float32),
        }
    )
    away_rows = pd.DataFrame(
        {
            "match_idx": df.index,
            "date": df["date"],
            "team": df["away_team"],
            "is_home": 0,
            "gf": df["away_score"].astype(np.float32),
            "ga": df["home_score"].astype(np.float32),
        }
    )

    long_df = pd.concat([home_rows, away_rows], ignore_index=True)
    long_df = long_df.sort_values(["team", "date", "match_idx"]).reset_index(drop=True)

    # Rest days feature: days since last match of the team (pre-match).
    long_df["prev_date"] = long_df.groupby("team", sort=False)["date"].shift(1)
    long_df["rest_days"] = (
        (long_df["date"] - long_df["prev_date"]).dt.total_seconds() / 86400.0
    ).astype(np.float32)
    long_df.loc[long_df["rest_days"].isna(), "rest_days"] = np.nan

    team_group = long_df.groupby("team", sort=False)

    for name, window in ("l5", cfg.l5), ("l10", cfg.l10):
        long_df[f"avg_gf_{name}"] = team_group["gf"].transform(
            lambda s: s.rolling(window=window, min_periods=1).mean().shift(1)
        )
        long_df[f"avg_ga_{name}"] = team_group["ga"].transform(
            lambda s: s.rolling(window=window, min_periods=1).mean().shift(1)
        )
        long_df[f"avg_gd_{name}"] = team_group["gf"].transform(
            lambda s: s.rolling(window=window, min_periods=1).mean().shift(1)
        ) - team_group["ga"].transform(
            lambda s: s.rolling(window=window, min_periods=1).mean().shift(1)
        )

    # Home-only and away-only
    for side, mask in ("home", long_df["is_home"] == 1), (
        "away",
        long_df["is_home"] == 0,
    ):
        sub = long_df.loc[mask, ["team", "date", "match_idx", "gf", "ga"]].copy()
        sub = sub.sort_values(["team", "date", "match_idx"]).reset_index(drop=True)
        g = sub.groupby("team", sort=False)
        sub[f"avg_gf_{side}_l10"] = g["gf"].transform(
            lambda s: s.rolling(window=cfg.l10, min_periods=1).mean().shift(1)
        )
        sub[f"avg_ga_{side}_l10"] = g["ga"].transform(
            lambda s: s.rolling(window=cfg.l10, min_periods=1).mean().shift(1)
        )
        sub[f"avg_gd_{side}_l10"] = g["gf"].transform(
            lambda s: s.rolling(window=cfg.l10, min_periods=1).mean().shift(1)
        ) - g["ga"].transform(
            lambda s: s.rolling(window=cfg.l10, min_periods=1).mean().shift(1)
        )

        long_df[f"avg_gf_{side}_l10"] = np.nan
        long_df[f"avg_ga_{side}_l10"] = np.nan
        long_df[f"avg_gd_{side}_l10"] = np.nan
        long_df.loc[mask, f"avg_gf_{side}_l10"] = sub[f"avg_gf_{side}_l10"].to_numpy()
        long_df.loc[mask, f"avg_ga_{side}_l10"] = sub[f"avg_ga_{side}_l10"].to_numpy()
        long_df.loc[mask, f"avg_gd_{side}_l10"] = sub[f"avg_gd_{side}_l10"].to_numpy()

    # Split home/away role back
    home_feat = long_df.loc[long_df["is_home"] == 1].set_index("match_idx")
    away_feat = long_df.loc[long_df["is_home"] == 0].set_index("match_idx")

    df["home_rest_days"] = home_feat["rest_days"].reindex(df.index).to_numpy()
    df["away_rest_days"] = away_feat["rest_days"].reindex(df.index).to_numpy()

    df["home_avg_gf_l5"] = home_feat["avg_gf_l5"].reindex(df.index).to_numpy()
    df["home_avg_ga_l5"] = home_feat["avg_ga_l5"].reindex(df.index).to_numpy()
    df["home_avg_gf_l10"] = home_feat["avg_gf_l10"].reindex(df.index).to_numpy()
    df["home_avg_ga_l10"] = home_feat["avg_ga_l10"].reindex(df.index).to_numpy()
    df["home_avg_gd_l5"] = home_feat["avg_gd_l5"].reindex(df.index).to_numpy()
    df["home_avg_gd_l10"] = home_feat["avg_gd_l10"].reindex(df.index).to_numpy()
    df["home_avg_gf_home_l10"] = (
        home_feat["avg_gf_home_l10"].reindex(df.index).to_numpy()
    )
    df["home_avg_ga_home_l10"] = (
        home_feat["avg_ga_home_l10"].reindex(df.index).to_numpy()
    )
    df["home_avg_gd_home_l10"] = (
        home_feat["avg_gd_home_l10"].reindex(df.index).to_numpy()
    )

    df["away_avg_gf_l5"] = away_feat["avg_gf_l5"].reindex(df.index).to_numpy()
    df["away_avg_ga_l5"] = away_feat["avg_ga_l5"].reindex(df.index).to_numpy()
    df["away_avg_gf_l10"] = away_feat["avg_gf_l10"].reindex(df.index).to_numpy()
    df["away_avg_ga_l10"] = away_feat["avg_ga_l10"].reindex(df.index).to_numpy()
    df["away_avg_gf_away_l10"] = (
        away_feat["avg_gf_away_l10"].reindex(df.index).to_numpy()
    )
    df["away_avg_ga_away_l10"] = (
        away_feat["avg_ga_away_l10"].reindex(df.index).to_numpy()
    )

    df["away_avg_gd_l5"] = df["away_avg_gf_l5"] - df["away_avg_ga_l5"]
    df["away_avg_gd_l10"] = df["away_avg_gf_l10"] - df["away_avg_ga_l10"]
    df["away_avg_gd_away_l10"] = df["away_avg_gf_away_l10"] - df["away_avg_ga_away_l10"]

    # Fill initial NaNs using global means (train split will re-standardize later).
    # Use an explicit numeric feature list to avoid accidentally casting string columns.
    numeric_cols: list[str] = [
        "month",
        "weekday",
        "season_start_year",
        "home_rest_days",
        "away_rest_days",
        "odds_home",
        "odds_draw",
        "odds_away",
        "odds_overround",
        "prob_home",
        "prob_draw",
        "prob_away",
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
    ]

    for c in numeric_cols:
        if c not in df.columns:
            raise KeyError(f"Missing expected feature column: {c}")
        df[c] = df[c].astype(np.float32)
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].mean())

    return df
