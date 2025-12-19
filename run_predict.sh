#!/usr/bin/env bash
set -euo pipefail

# -------- Config (edit here) --------
MODEL_PATH="artifacts/model.pt"
ODDS_MODEL_PATH="artifacts/odds.pt"
DATA_PATH="data/processed/matches.csv"
DEVICE="auto"

# -------- Match (edit here) --------

# Example 1: Real match but not in dataset
LEAGUE="Bundesliga"
HOME_TEAM="M'gladbach"
AWAY_TEAM="Dortmund"
DATE="2025-12-19T19:30:00Z"

# Example 2: Real match in dataset
# LEAGUE="Bundesliga"
# HOME_TEAM="Dortmund"
# AWAY_TEAM="Hoffenheim"
# DATE="2025-12-07T16:30:00Z"

# Example 3: Real match but not in dataset and no Dortmund
# LEAGUE="Bundesliga"
# HOME_TEAM="Bayern Munich"
# AWAY_TEAM="Mainz"
# DATE="2025-12-14T16:30:00Z"

# Example 4: Imaginary match and big scores are expected
# LEAGUE="Bundesliga"
# HOME_TEAM="Bayern Munich"
# AWAY_TEAM="Heidenheim"
# DATE="2025-12-14T16:30:00Z"

# Optional odds (set to empty to omit)
ODDS_HOME=""
ODDS_DRAW=""
ODDS_AWAY=""

MAX_GOALS=6
TOPK=10

# -------- Build command --------
CMD=(uv run scripts/predict.py
	--model "${MODEL_PATH}"
	--odds-model "${ODDS_MODEL_PATH}"
	--data "${DATA_PATH}"
	--device "${DEVICE}"
	--league "${LEAGUE}"
	--home "${HOME_TEAM}"
	--away "${AWAY_TEAM}"
	--date "${DATE}"
	--max-goals "${MAX_GOALS}"
	--topk "${TOPK}")

if [[ -n "${ODDS_HOME}" ]]; then
	CMD+=(--odds-home "${ODDS_HOME}")
fi
if [[ -n "${ODDS_DRAW}" ]]; then
	CMD+=(--odds-draw "${ODDS_DRAW}")
fi
if [[ -n "${ODDS_AWAY}" ]]; then
	CMD+=(--odds-away "${ODDS_AWAY}")
fi

# -------- Run --------
"${CMD[@]}"
