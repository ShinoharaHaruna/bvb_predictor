from __future__ import annotations

import argparse
import json
import re
from datetime import timedelta, timezone
from pathlib import Path
from typing import Final

import pandas as pd
from zoneinfo import ZoneInfo

from bvb_predictor.pipeline import PredictRequest, predict_match

_TZ_OFFSET_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:UTC)?\s*([+-])\s*(\d{1,2})(?::?(\d{2}))?$", re.IGNORECASE
)


def _timezone_arg(value: str) -> timezone | ZoneInfo:
    value = value.strip()
    aliases = {
        "Z": "UTC",
        "UTC": "UTC",
        "CST": "Asia/Shanghai",  # China Standard Time
    }

    mapped = aliases.get(value.upper())
    if mapped:
        value = mapped

    if value.upper() == "UTC":
        return timezone.utc

    m = _TZ_OFFSET_RE.match(value)
    if m:
        sign, hour_s, minute_s = m.groups()
        hours = int(hour_s)
        minutes = int(minute_s) if minute_s else 0
        delta = timedelta(hours=hours, minutes=minutes)
        if sign == "-":
            delta = -delta
        return timezone(delta)

    try:
        return ZoneInfo(value)
    except Exception as exc:  # pragma: no cover - argparse handles presentation
        raise argparse.ArgumentTypeError(f"Unsupported timezone: {value}") from exc


def _normalize_kickoff(dt_str: str, tzinfo: timezone | ZoneInfo) -> str:
    ts = pd.to_datetime(dt_str, utc=False, errors="raise")

    if ts.tzinfo is None:
        ts = ts.tz_localize(tzinfo)

    ts_utc = ts.tz_convert("UTC")
    return ts_utc.strftime("%Y-%m-%dT%H:%M:%SZ")


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
        "--date-tz",
        type=_timezone_arg,
        default=_timezone_arg("UTC"),
        help="Timezone of the provided --date. Accepts tz database names (e.g. Asia/Shanghai) or UTC offsets like +08:00.",
    )
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

    kickoff_iso = _normalize_kickoff(args.date, args.date_tz)

    req = PredictRequest(
        model_path=Path(args.model),
        data_path=Path(args.data),
        league=args.league,
        home_team=args.home,
        away_team=args.away,
        kickoff_iso=kickoff_iso,
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
