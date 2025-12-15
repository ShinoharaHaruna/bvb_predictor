from __future__ import annotations

import torch
from torch import nn


class OddsProbModel(nn.Module):
    def __init__(
        self,
        num_numerical_features: int,
        num_teams: int,
        num_leagues: int,
        team_embedding_dim: int = 12,
        league_embedding_dim: int = 4,
        hidden_dims: tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.team_emb = nn.Embedding(num_teams, team_embedding_dim)
        self.league_emb = nn.Embedding(num_leagues, league_embedding_dim)

        in_dim = num_numerical_features + team_embedding_dim * 2 + league_embedding_dim

        layers: list[nn.Module] = []
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(p=dropout))
            in_dim = h

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, 3)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        home = self.team_emb(x_cat[:, 0])
        away = self.team_emb(x_cat[:, 1])
        league = self.league_emb(x_cat[:, 2])

        x = torch.cat([x_num, home, away, league], dim=1)
        x = self.backbone(x)
        return self.head(x)
