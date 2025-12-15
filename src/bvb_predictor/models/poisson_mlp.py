from __future__ import annotations

import torch
from torch import nn


class TeamPoissonScoreModel(nn.Module):
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
        self.head = nn.Linear(in_dim, 2)
        self.softplus = nn.Softplus()

        self._league_rho = nn.Embedding(num_leagues, 1)
        self._league_alpha = nn.Embedding(num_leagues, 1)

    def forward(
        self, x_num: torch.Tensor, x_cat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x_cat: [B, 3] -> home_team_id, away_team_id, league_id
        home = self.team_emb(x_cat[:, 0])
        away = self.team_emb(x_cat[:, 1])
        league = self.league_emb(x_cat[:, 2])

        x = torch.cat([x_num, home, away, league], dim=1)
        x = self.backbone(x)
        lambdas = self.softplus(self.head(x))

        rho = torch.tanh(self._league_rho(x_cat[:, 2]).squeeze(1))
        alpha = self.softplus(self._league_alpha(x_cat[:, 2]).squeeze(1))
        return lambdas, rho, alpha


def poisson_nll(
    y_true: torch.Tensor, lambdas: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    lam = torch.clamp(lambdas, min=eps)
    return (lam - y_true * torch.log(lam)).mean()


def _nb_log_pmf(
    y: torch.Tensor, mu: torch.Tensor, alpha: torch.Tensor, eps: float
) -> torch.Tensor:
    mu = torch.clamp(mu, min=eps)
    alpha = torch.clamp(alpha, min=eps)

    r = 1.0 / alpha
    p = r / (r + mu)

    y = torch.clamp(y, min=0.0)
    return (
        torch.lgamma(y + r)
        - torch.lgamma(r)
        - torch.lgamma(y + 1.0)
        + r * torch.log(p)
        + y * torch.log(torch.clamp(1.0 - p, min=eps))
    )


def _dc_log_tau(
    y_home: torch.Tensor,
    y_away: torch.Tensor,
    mu_home: torch.Tensor,
    mu_away: torch.Tensor,
    rho: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    yh = y_home
    ya = y_away

    tau = torch.ones_like(rho)

    mask00 = (yh == 0) & (ya == 0)
    mask01 = (yh == 0) & (ya == 1)
    mask10 = (yh == 1) & (ya == 0)
    mask11 = (yh == 1) & (ya == 1)

    tau = torch.where(mask00, 1.0 - (mu_home * mu_away * rho), tau)
    tau = torch.where(mask01, 1.0 + (mu_home * rho), tau)
    tau = torch.where(mask10, 1.0 + (mu_away * rho), tau)
    tau = torch.where(mask11, 1.0 - rho, tau)

    tau = torch.clamp(tau, min=eps)
    return torch.log(tau)


def nb_dc_nll(
    y_true: torch.Tensor,
    mu: torch.Tensor,
    rho: torch.Tensor,
    alpha: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    y_home = y_true[:, 0]
    y_away = y_true[:, 1]

    mu_home = mu[:, 0]
    mu_away = mu[:, 1]

    log_p_home = _nb_log_pmf(y_home, mu_home, alpha, eps=eps)
    log_p_away = _nb_log_pmf(y_away, mu_away, alpha, eps=eps)
    log_tau = _dc_log_tau(y_home, y_away, mu_home, mu_away, rho, eps=eps)

    return -(log_p_home + log_p_away + log_tau).mean()


def nb_dc_nll_weighted(
    y_true: torch.Tensor,
    mu: torch.Tensor,
    rho: torch.Tensor,
    alpha: torch.Tensor,
    sample_weight: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    y_home = y_true[:, 0]
    y_away = y_true[:, 1]

    mu_home = mu[:, 0]
    mu_away = mu[:, 1]

    log_p_home = _nb_log_pmf(y_home, mu_home, alpha, eps=eps)
    log_p_away = _nb_log_pmf(y_away, mu_away, alpha, eps=eps)
    log_tau = _dc_log_tau(y_home, y_away, mu_home, mu_away, rho, eps=eps)
    per_row = -(log_p_home + log_p_away + log_tau)

    w = torch.clamp(sample_weight, min=0.0)
    denom = torch.clamp(w.sum(), min=eps)
    return (per_row * w).sum() / denom


def poisson_nll_weighted(
    y_true: torch.Tensor,
    lambdas: torch.Tensor,
    sample_weight: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    lam = torch.clamp(lambdas, min=eps)
    per_row = (lam - y_true * torch.log(lam)).sum(dim=1)
    w = torch.clamp(sample_weight, min=0.0)
    denom = torch.clamp(w.sum(), min=eps)
    return (per_row * w).sum() / denom
