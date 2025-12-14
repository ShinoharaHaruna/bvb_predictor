from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class LeagueEncoders:
    team_to_id: dict[str, int]
    league_to_id: dict[str, int]


def fit_league_encoders(df: pd.DataFrame) -> LeagueEncoders:
    teams = sorted(
        pd.unique(
            pd.concat([df["home_team"], df["away_team"]], ignore_index=True)
        ).tolist()
    )
    leagues = sorted(df["league"].unique().tolist())

    team_to_id = {name: i for i, name in enumerate(teams)}
    league_to_id = {name: i for i, name in enumerate(leagues)}
    return LeagueEncoders(team_to_id=team_to_id, league_to_id=league_to_id)


def transform_league_categoricals(
    df: pd.DataFrame, encoders: LeagueEncoders
) -> pd.DataFrame:
    out = df.copy()
    out["home_team_id"] = out["home_team"].map(encoders.team_to_id)
    out["away_team_id"] = out["away_team"].map(encoders.team_to_id)
    out["league_id"] = out["league"].map(encoders.league_to_id)

    if out["home_team_id"].isna().any() or out["away_team_id"].isna().any():
        raise ValueError("Found unseen team in transform.")
    if out["league_id"].isna().any():
        raise ValueError("Found unseen league in transform.")

    out["home_team_id"] = out["home_team_id"].astype(np.int64)
    out["away_team_id"] = out["away_team_id"].astype(np.int64)
    out["league_id"] = out["league_id"].astype(np.int64)
    return out


class LeagueScoreDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
):
    def __init__(self, df: pd.DataFrame, feature_cols: list[str]) -> None:
        self._df = df.reset_index(drop=True)
        self._feature_cols = feature_cols

        self._x_num = torch.tensor(
            self._df[self._feature_cols].to_numpy(dtype=np.float32), dtype=torch.float32
        )
        self._home_team_id = torch.tensor(
            self._df["home_team_id"].to_numpy(dtype=np.int64), dtype=torch.long
        )
        self._away_team_id = torch.tensor(
            self._df["away_team_id"].to_numpy(dtype=np.int64), dtype=torch.long
        )
        self._league_id = torch.tensor(
            self._df["league_id"].to_numpy(dtype=np.int64), dtype=torch.long
        )

        self._y_home = torch.tensor(
            self._df["home_score"].to_numpy(dtype=np.float32), dtype=torch.float32
        )
        self._y_away = torch.tensor(
            self._df["away_score"].to_numpy(dtype=np.float32), dtype=torch.float32
        )

        if "sample_weight" in self._df.columns:
            self._w = torch.tensor(
                self._df["sample_weight"].to_numpy(dtype=np.float32),
                dtype=torch.float32,
            )
        else:
            self._w = torch.ones(len(self._df), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self._x_num[idx]
        cat = torch.stack(
            [self._home_team_id[idx], self._away_team_id[idx], self._league_id[idx]],
            dim=0,
        )
        y = torch.stack([self._y_home[idx], self._y_away[idx]], dim=0)
        return x, cat, y, self._w[idx]
