from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MatchColumns:
    date: str = "date"
    league: str = "league"
    home_team: str = "home_team"
    away_team: str = "away_team"
    home_score: str = "home_score"
    away_score: str = "away_score"


MATCH_COLUMNS = MatchColumns()


REQUIRED_COLUMNS: tuple[str, ...] = (
    MATCH_COLUMNS.date,
    MATCH_COLUMNS.league,
    MATCH_COLUMNS.home_team,
    MATCH_COLUMNS.away_team,
    MATCH_COLUMNS.home_score,
    MATCH_COLUMNS.away_score,
)
