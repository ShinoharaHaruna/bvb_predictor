#!/usr/bin/env bash
set -euo pipefail

# -------- Config (edit here) --------
DATA_PATH="data/processed/matches.csv"
ARTIFACTS_DIR="artifacts"

VAL_RATIO=0.15
TEST_RATIO=0.15

EPOCHS=200
BATCH_SIZE=64
LR=0.001
PATIENCE=20
SEED=42

# Metrics settings
MAX_GOALS=6
TOPK=5

# Time weighting + finetune
TIME_DECAY_TAU_DAYS=365
FINETUNE_RECENT_SEASONS=3
FINETUNE_EPOCHS=30
FINETUNE_LR=0.0003

# Optional: memory optimization
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}

# -------- Run --------
uv run scripts/train.py \
	--data "${DATA_PATH}" \
	--artifacts "${ARTIFACTS_DIR}" \
	--val-ratio "${VAL_RATIO}" \
	--test-ratio "${TEST_RATIO}" \
	--epochs "${EPOCHS}" \
	--batch-size "${BATCH_SIZE}" \
	--lr "${LR}" \
	--patience "${PATIENCE}" \
	--seed "${SEED}" \
	--max-goals "${MAX_GOALS}" \
	--topk "${TOPK}" \
	--time-decay-tau-days "${TIME_DECAY_TAU_DAYS}" \
	--finetune-recent-seasons "${FINETUNE_RECENT_SEASONS}" \
	--finetune-epochs "${FINETUNE_EPOCHS}" \
	--finetune-lr "${FINETUNE_LR}"
