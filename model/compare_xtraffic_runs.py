"""Check two completed Alameda runs and report the paired test difference."""

import argparse
import json
from pathlib import Path


def read_summary(path):
    return json.loads(path.read_text(encoding="utf-8"))


def comparable_protocol(report):
    # The event encoder lives in the loader, which may differ across branches.
    return {key: value for key, value in report["protocol"].items()
            if key != "loader_source_sha256"}


def compare(m0, m1):
    if (m0["model"], m1["model"]) != ("m0", "m1"):
        raise ValueError("Expected an M0 report followed by an M1 report")
    if comparable_protocol(m0) != comparable_protocol(m1):
        raise ValueError("The reports have different data, graph or training settings")
    if m0["initial_backbone_sha256"] != m1["initial_backbone_sha256"]:
        raise ValueError("The initial backbone fingerprints differ")

    overlapping_epochs = min(len(m0["epochs"]), len(m1["epochs"]))
    if overlapping_epochs == 0:
        raise ValueError("Both reports must contain completed training epochs")
    for left, right in zip(m0["epochs"], m1["epochs"]):
        if (left["epoch"], left["train_order_sha256"]) != (right["epoch"], right["train_order_sha256"]):
            raise ValueError(f"Training order differs at epoch {left['epoch']}")

    m0_test, m1_test = m0["test"], m1["test"]
    if m0_test["valid_targets"] != m1_test["valid_targets"]:
        raise ValueError("The runs used different numbers of valid test targets")

    return {
        "seed": m0["protocol"]["seed"],
        "initial_backbone_sha256": m0["initial_backbone_sha256"],
        "overlapping_epochs_verified": overlapping_epochs,
        "m0_test": m0_test,
        "m1_test": m1_test,
        "m1_minus_m0": {metric: m1_test[metric] - m0_test[metric] for metric in ("mae", "rmse")},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m0-summary", type=Path, required=True)
    parser.add_argument("--m1-summary", type=Path, required=True)
    args = parser.parse_args()
    result = compare(read_summary(args.m0_summary), read_summary(args.m1_summary))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
