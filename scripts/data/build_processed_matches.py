from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProcessConfig:
    raw_dir: Path
    out_csv: Path
    out_report: Path
    league_code: str


def _infer_season_from_filename(path: Path) -> str:
    m = re.search(r"_(\d{4})\.csv$", path.name)
    return m.group(1) if m else "unknown"


def _match_id(league: str, season: str, dt: str, home: str, away: str) -> str:
    raw = f"{league}|{season}|{dt}|{home}|{away}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _pick_first(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _normalize_team_name(name: str) -> str:
    # Keep minimal, you can extend later with a mapping table.
    name = name.strip()
    name = re.sub(r"\s+", " ", name)
    return name


def _parse_datetime_series(date_raw: pd.Series, has_time: bool) -> pd.Series:
    date_raw = date_raw.astype(str)

    if has_time:
        formats = ["%d/%m/%y %H:%M", "%d/%m/%Y %H:%M"]
    else:
        formats = ["%d/%m/%y", "%d/%m/%Y"]

    dt: pd.Series | None = None
    for fmt in formats:
        parsed = pd.to_datetime(date_raw, errors="coerce", format=fmt, utc=True)
        if dt is None:
            dt = parsed
        else:
            dt = dt.fillna(parsed)

        if dt.notna().all():
            break

    if dt is None:
        dt = pd.to_datetime(date_raw, errors="coerce", utc=True)

    return dt


def _detect_encoding(path: Path) -> str:
    # Detect encoding without using pandas, since malformed CSV may fail parsing
    # before we have a chance to repair it.
    encodings = ["utf-8", "cp1252", "latin1"]
    last_err: Exception | None = None
    raw = path.read_bytes()[:8192]
    for enc in encodings:
        try:
            raw.decode(enc, errors="strict")
            return enc
        except UnicodeDecodeError as e:
            last_err = e

    raise RuntimeError(f"Failed to detect encoding for {path.name}: {last_err}")


def _validate_csv_shape(path: Path, encoding: str, repair: bool) -> None:
    # Validate raw CSV to pinpoint malformed lines instead of silently skipping.
    # If repair is enabled, we only fix trivial shape issues:
    # - An extra trailing empty field (line ends with ",")
    # - Missing trailing fields (line ends early), we pad with empty strings
    # - Extra trailing empty fields (multiple ending commas), we trim empty tail columns
    with path.open("r", encoding=encoding, errors="replace", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty csv file: {path}")
        expected = len(header)

        bad: list[tuple[int, int, list[str]]] = []
        for line_no, row in enumerate(reader, start=2):
            if len(row) != expected:
                bad.append((line_no, len(row), row))
                break

    if not bad:
        return

    line_no, got, row = bad[0]
    if repair:
        tmp = path.with_suffix(path.suffix + ".repaired")
        with (
            path.open("r", encoding=encoding, errors="replace", newline="") as rf,
            tmp.open("w", encoding=encoding, errors="replace", newline="") as wf,
        ):
            r = csv.reader(rf)
            w = csv.writer(wf)
            hdr = next(r)
            w.writerow(hdr)
            for row_no, rr in enumerate(r, start=2):
                if len(rr) > expected:
                    extra = rr[expected:]
                    if all((c == "" or c.isspace()) for c in extra):
                        rr = rr[:expected]
                    else:
                        preview = ",".join(rr)[:240]
                        raise ValueError(
                            f"Malformed CSV: {path.name} line {row_no} expected {expected} fields, got {len(rr)}. "
                            f"Non-empty extra columns found. Row preview: {preview}"
                        )

                if len(rr) < expected:
                    rr = rr + [""] * (expected - len(rr))

                w.writerow(rr)

        tmp.replace(path)
        return

    preview = ",".join(row)
    preview = preview[:240]
    raise ValueError(
        f"Malformed CSV: {path.name} line {line_no} expected {expected} fields, got {got}. "
        f"Row preview: {preview}"
    )


def _load_one_csv(path: Path, league_code: str) -> pd.DataFrame:
    enc = _detect_encoding(path)
    df = pd.read_csv(
        path,
        encoding=enc,
        encoding_errors="replace",
        engine="python",
    )

    # football-data columns vary by era; map to a stable schema
    col_date = _pick_first(df, ["Date"])
    col_time = _pick_first(df, ["Time"])
    col_home = _pick_first(df, ["HomeTeam"])
    col_away = _pick_first(df, ["AwayTeam"])
    col_hg = _pick_first(df, ["FTHG", "HG"])
    col_ag = _pick_first(df, ["FTAG", "AG"])

    if not all([col_date, col_home, col_away, col_hg, col_ag]):
        raise ValueError(f"Missing required columns in {path.name}")

    date_raw = df[col_date].astype(str)
    if col_time:
        date_raw = date_raw + " " + df[col_time].astype(str)

    dt = _parse_datetime_series(date_raw, has_time=col_time is not None)

    out = pd.DataFrame(
        {
            "date": dt,
            "league": league_code,
            "season": _infer_season_from_filename(path),
            "home_team": df[col_home].astype(str).map(_normalize_team_name),
            "away_team": df[col_away].astype(str).map(_normalize_team_name),
            "home_score": pd.to_numeric(df[col_hg], errors="coerce"),
            "away_score": pd.to_numeric(df[col_ag], errors="coerce"),
        }
    )

    # Odds (optional)
    col_oh = _pick_first(df, ["B365H", "AvgH"])
    col_od = _pick_first(df, ["B365D", "AvgD"])
    col_oa = _pick_first(df, ["B365A", "AvgA"])

    if col_oh:
        out["odds_home"] = pd.to_numeric(df[col_oh], errors="coerce")
    else:
        out["odds_home"] = np.nan

    if col_od:
        out["odds_draw"] = pd.to_numeric(df[col_od], errors="coerce")
    else:
        out["odds_draw"] = np.nan

    if col_oa:
        out["odds_away"] = pd.to_numeric(df[col_oa], errors="coerce")
    else:
        out["odds_away"] = np.nan

    return out


def _build_report(df: pd.DataFrame) -> dict[str, Any]:
    report: dict[str, Any] = {}
    report["rows"] = int(len(df))
    report["seasons"] = sorted(df["season"].dropna().unique().tolist())
    report["teams"] = int(
        pd.unique(pd.concat([df["home_team"], df["away_team"]], ignore_index=True)).size
    )

    dates = df["date"].dropna()
    report["date_min"] = None if dates.empty else str(dates.min())
    report["date_max"] = None if dates.empty else str(dates.max())

    report["missing_rate"] = {
        c: float(df[c].isna().mean())
        for c in [
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "odds_home",
            "odds_draw",
            "odds_away",
        ]
    }

    report["duplicate_match_id"] = int(df["match_id"].duplicated().sum())
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw/football-data.co.uk/D1",
        help="Raw data directory (new default layout encodes data source + competition)",
    )
    parser.add_argument("--out", type=str, default="data/processed/matches.csv")
    parser.add_argument("--report", type=str, default="data/processed/report.json")
    parser.add_argument("--league", type=str, default="Bundesliga")
    parser.add_argument("--league-code", type=str, default="D1")
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Repair trivial malformed lines (extra trailing empty field) in-place before parsing",
    )

    args = parser.parse_args()

    cfg = ProcessConfig(
        raw_dir=Path(args.raw_dir),
        out_csv=Path(args.out),
        out_report=Path(args.report),
        league_code=args.league,
    )

    paths = sorted(cfg.raw_dir.glob(f"{args.league_code}_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No raw csv found in {cfg.raw_dir}")

    frames: list[pd.DataFrame] = []
    for p in paths:
        # Diagnose bad CSV rows early with a precise file+line error.
        enc = _detect_encoding(p)
        _validate_csv_shape(p, encoding=enc, repair=args.repair)
        frames.append(_load_one_csv(p, league_code=cfg.league_code))

    df = pd.concat(frames, ignore_index=True)

    df = df.dropna(
        subset=["date", "home_team", "away_team", "home_score", "away_score"]
    ).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # Normalize and build match_id
    iso = df["date"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    df["match_id"] = [
        _match_id(args.league, s, d, h, a)
        for s, d, h, a in zip(
            df["season"].astype(str),
            iso.astype(str),
            df["home_team"],
            df["away_team"],
            strict=False,
        )
    ]

    df["league"] = args.league

    # De-duplicate
    df = df.sort_values(["date", "match_id"]).drop_duplicates(
        subset=["match_id"], keep="first"
    )

    # Stable column order
    df = df[
        [
            "match_id",
            "date",
            "league",
            "season",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "odds_home",
            "odds_draw",
            "odds_away",
        ]
    ].sort_values("date")

    cfg.out_csv.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_report.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(cfg.out_csv, index=False)

    report = _build_report(df)
    report["source"] = "football-data.co.uk"
    report["competition_code"] = args.league_code
    report["competition_name"] = args.league
    cfg.out_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
