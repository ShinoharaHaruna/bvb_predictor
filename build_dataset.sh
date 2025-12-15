#!/usr/bin/env bash
set -euo pipefail

# Download raw data
uv run scripts/data/fetch_football_data.py --all --confirm

# Build dataset
uv run scripts/data/build_processed_matches.py --repair
