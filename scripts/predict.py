from __future__ import annotations

import argparse
import json
from pathlib import Path

from bvb_predictor.pipeline import PredictRequest, predict_match


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="artifacts/model.pt")
    parser.add_argument("--data", type=str, default="data/processed/matches.csv")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device: auto|cpu|cuda|cuda:0 ...",
    )
    parser.add_argument("--league", type=str, required=True)
    parser.add_argument("--home", type=str, required=True)
    parser.add_argument("--away", type=str, required=True)
    parser.add_argument("--date", type=str, required=True)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional runtime info to stderr.",
    )
    parser.add_argument("--odds-home", type=float, default=float("nan"))
    parser.add_argument("--odds-draw", type=float, default=float("nan"))
    parser.add_argument("--odds-away", type=float, default=float("nan"))
    parser.add_argument(
        "--odds-model",
        type=str,
        default="",
        help="Optional odds model checkpoint. If set and odds are omitted, use it to predict prob_home/draw/away.",
    )
    parser.add_argument(
        "--max-goals",
        type=int,
        default=-1,
        help="Score matrix cutoff. Negative means auto-select via model outputs.",
    )
    parser.add_argument("--topk", type=int, default=10)

    args = parser.parse_args()

    req = PredictRequest(
        model_path=Path(args.model),
        data_path=Path(args.data),
        league=args.league,
        home_team=args.home,
        away_team=args.away,
        kickoff_iso=args.date,
        device=args.device,
        odds_model_path=args.odds_model,
        odds_home=args.odds_home,
        odds_draw=args.odds_draw,
        odds_away=args.odds_away,
        max_goals=args.max_goals,
        topk=args.topk,
        verbose=args.verbose,
    )

    out = predict_match(req)
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
