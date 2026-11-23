"""Verify the fixed road graph constructed by the Tokyo training entry point."""

from pathlib import Path
import sys

import torch
import yaml

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODEL_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from model.train import build_training_model


def main():
    with (MODEL_DIR / "ConFormer.yaml").open(encoding="utf-8") as config_file:
        model_args = yaml.safe_load(config_file)["TKY"]["model_args"]

    model = build_training_model("TKY", PROJECT_DIR / "data" / "TKY", model_args)
    (fixed_graph,) = model.graph_propagate.supports
    expected_shape = (model_args["num_nodes"],) * 2
    if fixed_graph.shape != expected_shape:
        raise RuntimeError(f"Expected fixed graph shape {expected_shape}, got {tuple(fixed_graph.shape)}")

    column_sums = fixed_graph.sum(dim=0)
    min_sum, max_sum = torch.aminmax(column_sums)
    print(
        f"Fixed road graph shape: {tuple(fixed_graph.shape)}. "
        f"Column sum range: [{min_sum.item():.6g}, {max_sum.item():.6g}]"
    )
    if not torch.allclose(column_sums, torch.ones_like(column_sums)):
        raise RuntimeError("The model's fixed Tokyo road graph must have unit column sums")

    print("PASS: Tokyo training model loaded the incoming-normalized road graph.")


if __name__ == "__main__":
    main()
