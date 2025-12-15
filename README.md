# bvb_predictor

[View in Chinese](README_CN.md)

## Abstract

This repository implements a lightweight, reproducible pipeline for **probabilistically accurate score prediction** in football. We predict the goal distribution using a neural network with team/league embeddings and time-aware pre-match features, and output the complete score probability matrix (within a bounded grid) and its derived win/draw/loss probabilities.

Key design goals:

- **No look-ahead leakage**: all pre-match rolling statistics are computed with a one-step shift.
- **Non-stationarity handling**: recent matches are emphasized by time-decay weighting and an optional fine-tuning stage.
- **Practical inference**: produces a full score distribution (within a bounded grid) and derived W/D/L probabilities.

## Method

### Problem formulation

Given a match $(home, away, t)$, predict the expected goals $\mu_{home}$ and $\mu_{away}$. We then obtain a bounded exact-score distribution on a goal grid $[0,\dots,G]$.

In addition to the baseline independent Poisson score model, we support:

- **Negative Binomial (NB)** goal counts for over-dispersion (variance > mean)
- **Dixon-Coles (DC)** low-score correction for correlation between home/away goals

The current default scoring distribution used in training/inference is **NB + DC**, with **league-specific** learned parameters:

- $\rho_{league}$: Dixon-Coles correlation parameter
- $\alpha_{league}$: Negative Binomial dispersion parameter

#### NB+DC score distribution (bounded grid)

We use a Negative Binomial parameterization with mean $\mu$ and dispersion $\alpha$:

$$
P_{NB}(X=k\mid \mu,\alpha)=\frac{\Gamma(k+\alpha^{-1})}{\Gamma(\alpha^{-1})\,k!}\left(\frac{\alpha^{-1}}{\alpha^{-1}+\mu}\right)^{\alpha^{-1}}\left(\frac{\mu}{\alpha^{-1}+\mu}\right)^k.
$$

Dixon-Coles applies a low-score correction factor $\tau(h,a)$:

$$
\tau(h,a)=
\begin{cases}
1-\mu_{home}\mu_{away}\rho, & (h,a)=(0,0)\\
1+\mu_{home}\rho, & (h,a)=(0,1)\\
1+\mu_{away}\rho, & (h,a)=(1,0)\\
1-\rho, & (h,a)=(1,1)\\
1, & \text{otherwise.}
\end{cases}
$$

On the bounded grid $h,a\in[0,\dots,G]$, the joint distribution is defined as:

$$
P(H=h,A=a) \propto \tau(h,a)\,P_{NB}(H=h\mid\mu_{home},\alpha)\,P_{NB}(A=a\mid\mu_{away},\alpha),
$$

and is normalized over the grid.

### Model

The score model is implemented in `src/bvb_predictor/models/poisson_mlp.py` as `TeamPoissonScoreModel`:

- Team embeddings: `home_team_id`, `away_team_id`
- League embedding: `league_id`
- MLP backbone (ReLU + BatchNorm + Dropout)
- Two-headed `Softplus` output to ensure $\mu>0$
- Learnable per-league parameters $\rho$ (DC) and $\alpha$ (NB)

The training objective is the **NB+DC negative log-likelihood (NLL)**. When enabled, a **time-decay** weighted variant is used.

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

- Rolling pre-match team statistics (shifted): GF/GA/GD over L1/L3/L5, plus home-only / away-only splits.
- Exponential moving average (EMA) features for GF/GA/GD.
- Rest-day features: `home_rest_days`, `away_rest_days` (days since last match).
- Season feature: `season_start_year`.
- Odds-derived signals:
  - implied probabilities `prob_home/prob_draw/prob_away`
  - overround `odds_overround`
  - availability flag `odds_available`

The odds/prob features support two modes:

- If raw odds are present (`odds_home/draw/away`), probabilities are derived by normalization.
- If odds are missing but `prob_home/draw/away` is provided (e.g. predicted by an odds model), they are used as-is and treated as available.

All features are computed in chronological order and use a `shift(1)` to prevent future leakage.

## Training

### Script entry points

- Training: `scripts/train.py`
- Odds model training (optional): `scripts/train_odds.py`
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
- `artifacts/odds.pt` (optional odds model weights + encoders + normalization stats)
- `artifacts/metrics.json` (evaluation summary)

### Metrics

`scripts/train.py` reports:

- NB+DC NLL on validation/test
- MAE of $\mu$ vs realized goals
- W/D/L accuracy (derived from the predicted score matrix)
- exact-score top-1 and top-k hit rates

Training supports an optional **odds mix** strategy when an odds model is available:

- A fraction of training rows will replace `prob_home/draw/away` with the odds-model predicted probabilities.
- This reduces the train/inference mismatch when users do not provide real odds at inference time.

## Inference

Edit `run_predict.sh` and run:

```bash
./run_predict.sh
```

The predictor returns a JSON containing:

- `mu_home`, `mu_away`
- `rho`, `alpha`
- `topk_scores` (most probable scorelines within $[0,G]$)
- `wdl` (home/draw/away probabilities)

Inference supports **arbitrary match dates**: historical rows are automatically truncated to those strictly before the target date.

Inference supports an optional **odds model**:

- If the user omits raw odds and provides `--odds-model artifacts/odds.pt`, the script will predict `prob_home/draw/away` and use them as pseudo-odds features.

## GPU / device

All torch entrypoints support `--device`:

- `auto` (default): use CUDA if available
- `cpu`
- `cuda` / `cuda:0` / ...

The wrapper scripts also expose a `DEVICE` variable:

- `run_train.sh`: passes `--device` to both `train_odds.py` and `train.py`
- `run_predict.sh`: passes `--device` to `predict.py`

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
  train_odds.py
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
