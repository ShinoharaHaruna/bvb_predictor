#!/usr/bin/env bash
set -euo pipefail

# -------- Config (edit here) --------
MODEL_PATH="artifacts/model.pt"
DATA_PATH="data/processed/matches.csv"

LEAGUE="Bundesliga"
HOME_TEAM="Freiburg"
AWAY_TEAM="Dortmund"
DATE="2025-12-14T14:30:00Z"

# Optional odds (set to empty to omit)
ODDS_HOME=""
ODDS_DRAW=""
ODDS_AWAY=""

MAX_GOALS=6
TOPK=10

# -------- Build command --------
CMD=(uv run scripts/predict.py
	--model "${MODEL_PATH}"
	--data "${DATA_PATH}"
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
