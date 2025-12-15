from __future__ import annotations

import argparse
from pathlib import Path

import torch

from bvb_predictor.models.odds_mlp import OddsProbModel
from bvb_predictor.models.poisson_mlp import TeamPoissonScoreModel


def _export_score_model(ckpt_path: Path, out_path: Path, opset: int) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")

    encoders = ckpt["encoders"]
    feature_cols: list[str] = ckpt["feature_cols"]

    team_to_id: dict[str, int] = encoders["team_to_id"]
    league_to_id: dict[str, int] = encoders["league_to_id"]

    model = TeamPoissonScoreModel(
        num_numerical_features=len(feature_cols),
        num_teams=len(team_to_id),
        num_leagues=len(league_to_id),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    x_num = torch.zeros((1, len(feature_cols)), dtype=torch.float32)
    x_cat = torch.zeros((1, 3), dtype=torch.long)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (x_num, x_cat),
        str(out_path),
        input_names=["x_num", "x_cat"],
        output_names=["mu", "rho", "alpha"],
        dynamic_axes={
            "x_num": {0: "batch"},
            "x_cat": {0: "batch"},
            "mu": {0: "batch"},
            "rho": {0: "batch"},
            "alpha": {0: "batch"},
        },
        opset_version=opset,
    )


def _export_odds_model(ckpt_path: Path, out_path: Path, opset: int) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")

    encoders = ckpt["encoders"]
    feature_cols: list[str] = ckpt["feature_cols"]

    team_to_id: dict[str, int] = encoders["team_to_id"]
    league_to_id: dict[str, int] = encoders["league_to_id"]

    model = OddsProbModel(
        num_numerical_features=len(feature_cols),
        num_teams=len(team_to_id),
        num_leagues=len(league_to_id),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    x_num = torch.zeros((1, len(feature_cols)), dtype=torch.float32)
    x_cat = torch.zeros((1, 3), dtype=torch.long)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (x_num, x_cat),
        str(out_path),
        input_names=["x_num", "x_cat"],
        output_names=["logits"],
        dynamic_axes={
            "x_num": {0: "batch"},
            "x_cat": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=opset,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="artifacts/model.pt")
    parser.add_argument("--out", type=str, default="artifacts/model.onnx")
    parser.add_argument("--odds-model", type=str, default="")
    parser.add_argument("--odds-out", type=str, default="artifacts/odds.onnx")
    parser.add_argument("--opset", type=int, default=17)

    args = parser.parse_args()

    _export_score_model(Path(args.model), Path(args.out), opset=args.opset)

    if args.odds_model:
        _export_odds_model(Path(args.odds_model), Path(args.odds_out), opset=args.opset)


if __name__ == "__main__":
    main()
