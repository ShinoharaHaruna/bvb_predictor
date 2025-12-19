from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from tqdm import tqdm


BASE_URL = "https://www.football-data.co.uk/mmz4281/"


@dataclass(frozen=True)
class DownloadPlanItem:
    season: str
    league: str
    url: str
    out_path: str


def _http_get(url: str, timeout_s: int = 30) -> bytes:
    # football-data.co.uk may block non-browser requests on some networks.
    # Use a browser-like header set to reduce 403 chances.
    req = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.football-data.co.uk/",
            "Connection": "keep-alive",
        },
    )
    with urlopen(req, timeout=timeout_s) as resp:
        return resp.read()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _list_seasons_from_index() -> list[str]:
    # Index is a simple directory listing page (Apache-style).
    html = _http_get(BASE_URL).decode("utf-8", errors="ignore")

    # Typical pattern: href="2425/"
    seasons = sorted(set(re.findall(r'href="(\d{4})/"', html)))

    # Some pages may contain other 4-digit dirs; keep only plausible football-data seasons.
    # Empirically, seasons are like 0001 .. 2425.
    seasons = [s for s in seasons if s.isdigit()]
    return seasons


def _list_seasons_fallback() -> list[str]:
    # Fallback when directory listing is blocked.
    # football-data season code is usually YY(YY+1), e.g. 2425.
    # We generate a broad candidate range and let download stage filter via HTTP status.
    current_year = time.gmtime().tm_year
    max_yy = (current_year + 1) % 100

    seasons: list[str] = []
    for start in range(0, max_yy + 1):
        end = (start + 1) % 100
        seasons.append(f"{start:02d}{end:02d}")

    return seasons


def _build_plan(
    seasons: list[str], league: str, out_dir: Path
) -> list[DownloadPlanItem]:
    items: list[DownloadPlanItem] = []
    for season in seasons:
        url = f"{BASE_URL}{season}/{league}.csv"
        out_path = out_dir / f"{league}_{season}.csv"
        items.append(
            DownloadPlanItem(
                season=season,
                league=league,
                url=url,
                out_path=str(out_path),
            )
        )
    return items


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "generated_at": None, "items": []}

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def _print_plan(items: list[DownloadPlanItem]) -> None:
    for it in items:
        print(
            json.dumps(
                {"season": it.season, "url": it.url, "out": it.out_path},
                ensure_ascii=False,
            )
        )


def _download_with_retry(url: str, retries: int, sleep_s: float) -> bytes:
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return _http_get(url)
        except (HTTPError, URLError, TimeoutError) as e:
            last_err = e
            if attempt < retries:
                time.sleep(sleep_s)
                continue
            raise

    raise RuntimeError(f"unreachable: {last_err}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--league", type=str, default="D1", help="football-data league code, e.g. D1"
    )
    # Default path encodes data source + competition for future multi-source/multi-competition support.
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/raw/football-data.co.uk/D1",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/raw/football-data.co.uk/D1/manifest.json",
    )

    parser.add_argument(
        "--seasons",
        type=str,
        default="",
        help="Comma separated season codes, e.g. 2425,2324",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Plan for all available seasons (default: dry-run)",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually download files. Without it we only print plan.",
    )

    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore cache and re-download matching files.",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.seasons:
        seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    elif args.all:
        try:
            seasons = _list_seasons_from_index()
        except HTTPError as e:
            if e.code == 403:
                print(
                    "Index listing is blocked (HTTP 403). Falling back to probing candidate seasons.",
                    file=sys.stderr,
                )
                seasons = _list_seasons_fallback()
            else:
                raise
    else:
        print("Either --seasons or --all is required.", file=sys.stderr)
        raise SystemExit(2)

    plan = _build_plan(seasons=seasons, league=args.league, out_dir=out_dir)

    # Always show plan for controllability.
    _print_plan(plan)

    if not args.confirm:
        print("Dry-run only. Re-run with --confirm to download.", file=sys.stderr)
        return

    manifest_path = Path(args.manifest)
    manifest = _load_manifest(manifest_path)
    manifest["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    items_out: list[dict[str, Any]] = []
    existing_by_out = {it.get("out_path"): it for it in manifest.get("items", [])}

    for it in tqdm(plan, desc="Downloading", unit="file"):
        out_path = Path(it.out_path)

        # Cache: if file exists and manifest contains sha256, we skip.
        prev = existing_by_out.get(str(out_path))
        if not args.force_refresh and out_path.exists() and prev and prev.get("sha256"):
            print(f"[cache] Skip {out_path}", file=sys.stderr)
            items_out.append(prev)
            continue

        try:
            data = _download_with_retry(
                it.url, retries=args.retries, sleep_s=args.sleep
            )
        except HTTPError as e:
            # Some seasons may not have the league file; skip but record.
            items_out.append(
                {
                    "season": it.season,
                    "league": it.league,
                    "url": it.url,
                    "out_path": str(out_path),
                    "status": f"http_error_{e.code}",
                }
            )
            continue

        sha = _sha256(data)
        out_path.write_bytes(data)

        items_out.append(
            {
                "season": it.season,
                "league": it.league,
                "url": it.url,
                "out_path": str(out_path),
                "sha256": sha,
                "status": "downloaded",
            }
        )

    manifest["items"] = items_out
    _save_manifest(manifest_path, manifest)


if __name__ == "__main__":
    main()
