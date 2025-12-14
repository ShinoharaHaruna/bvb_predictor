# bvb_predictor

[View in Chinese](README_CN.md)

## Abstract

This repository implements a lightweight, reproducible pipeline for **probabilistic exact score prediction** in football. We model home- and away-goals as conditionally independent Poisson variables with rates $\lambda_{home}$ and $\lambda_{away}$, predicted by a neural network with team embeddings and time-aware pre-match features.

Key design goals:

- **No look-ahead leakage**: all pre-match rolling statistics are computed with a one-step shift.
- **Non-stationarity handling**: recent matches are emphasized by time-decay weighting and an optional fine-tuning stage.
- **Practical inference**: produces a full score distribution (within a bounded grid) and derived W/D/L probabilities.

## Method

### Problem formulation

Given a match $(home, away, t)$, predict $\lambda_{home}$ and $\lambda_{away}$. We then obtain the score distribution:

$$
P(H=h, A=a) = \text{Poisson}(h;\lambda_{home})\;\text{Poisson}(a;\lambda_{away}),\quad h,a \in [0,\dots,G].
$$

### Model

The model is implemented in `src/bvb_predictor/models/poisson_mlp.py` as `TeamPoissonScoreModel`:

- Team embeddings: `home_team_id`, `away_team_id`
- League embedding: `league_id`
- MLP backbone (ReLU + BatchNorm + Dropout)
- Two-headed `Softplus` output to ensure $\lambda>0$

The training objective is Poisson negative log-likelihood (NLL). When enabled, a **time-decay** weighted variant is used.

### Time-aware learning (A+B+C)

To reflect that football performance drifts over time:

- **(A) Time-decay weighting**: assign sample weight $w=\exp(-\Delta days/\tau)$ on the *training split*.
- **(B) Fine-tuning**: after the main training stage, optionally fine-tune on the last $N$ seasons within the training split.
- **(C) Explicit time features**: include season and rest-day signals.

## Data

### Raw acquisition

We currently support `football-data.co.uk` Bundesliga (`D1`) in `scripts/data/fetch_football_data.py`.

Raw files are stored under:

```text
data/raw/football-data.co.uk/D1/
```

### Cleaning and standardization

The processing script `scripts/data/build_processed_matches.py` builds:

```text
data/processed/matches.csv
data/processed/report.json
```

The processed schema (minimum) is:

```text
match_id,date,league,season,home_team,away_team,home_score,away_score,odds_home,odds_draw,odds_away
```

Notes:

- The script includes a `--repair` mode to normalize malformed CSV rows (padding/trimming trailing empty columns) while preserving match rows.

## Feature engineering

`src/bvb_predictor/features/league_rolling.py` implements `build_league_features`:

- Rolling pre-match team statistics (shifted): GF/GA/GD over L5/L10, plus home-only / away-only L10.
- Rest-day features: `home_rest_days`, `away_rest_days` (days since last match).
- Season feature: `season_start_year`.
- Odds-derived signals:
  - implied probabilities `prob_home/prob_draw/prob_away`
  - overround `odds_overround`
  - availability flag `odds_available`

All features are computed in chronological order and use a `shift(1)` to prevent future leakage.

## Training

### Script entry points

- Training: `scripts/train.py`
- Inference: `scripts/predict.py`
- Convenience wrappers:
  - `run_train.sh`
  - `run_predict.sh`

### Running training

Edit `run_train.sh` to set paths and hyperparameters, then run:

```bash
./run_train.sh
```

Artifacts:

- `artifacts/model.pt` (model weights + encoders + normalization stats)
- `artifacts/metrics.json` (evaluation summary)

### Metrics

`scripts/train.py` reports:

- Poisson NLL on validation/test
- MAE of $\lambda$ vs realized goals
- W/D/L accuracy (derived from the predicted score matrix)
- exact-score top-1 and top-k hit rates

## Inference

Edit `run_predict.sh` and run:

```bash
./run_predict.sh
```

The predictor returns a JSON containing:

- `lambda_home`, `lambda_away`
- `topk_scores` (most probable scorelines within $[0,G]$)
- `wdl` (home/draw/away probabilities)

Inference supports **arbitrary match dates**: historical rows are automatically truncated to those strictly before the target date.

## Repository structure

```text
src/bvb_predictor/
  data/
    dataset.py
  features/
    league_rolling.py
  models/
    poisson_mlp.py
  utils/
    score_prob.py

scripts/
  train.py
  predict.py
  data/
    fetch_football_data.py
    build_processed_matches.py

run_train.sh
run_predict.sh
```

## Limitations and future work

- The Poisson independence assumption may underestimate correlation between home/away goals.
- Feature set is intentionally minimal; incorporating richer covariates (xG, lineups, injuries) should improve calibration.
- Data source coverage currently focuses on Bundesliga; extending to European competitions (UCL/UEL) and multi-league training is a natural next step.
