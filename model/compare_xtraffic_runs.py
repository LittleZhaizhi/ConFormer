"""Validate a paired Alameda run and summarize archived test predictions."""

import argparse
import json
from pathlib import Path

import numpy as np


EVENT_CATEGORIES = {
    1: "hazard",
    2: "other",
    3: "accident",
    4: "breakdown",
    5: "police",
    6: "weather",
}


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
        "m1_minus_m0": {name: m1_test[name] - m0_test[name] for name in ("mae", "rmse")},
    }


def load_predictions(path):
    required = {
        "sample_ids", "predictions", "targets", "valid_mask",
        "event_mask", "event_type_ids",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        arrays = {name: np.asarray(archive[name]) for name in required}

    predictions, targets = arrays["predictions"], arrays["targets"]
    if predictions.ndim == 4 and predictions.shape[-1] == 1:
        predictions = predictions[..., 0]
    if targets.ndim == 4 and targets.shape[-1] == 1:
        targets = targets[..., 0]
    if predictions.ndim != 3 or predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have matching (samples, horizons, nodes) shapes")

    sample_count, _, node_count = predictions.shape
    valid, event = arrays["valid_mask"], arrays["event_mask"]
    sample_ids, categories = arrays["sample_ids"], arrays["event_type_ids"]
    if valid.ndim == 4 and valid.shape[-1] == 1:
        valid = valid[..., 0]
    if valid.shape != predictions.shape:
        raise ValueError("valid_mask must match prediction positions")
    if event.shape != (sample_count, node_count):
        raise ValueError("event_mask must have shape (samples, nodes)")
    if sample_ids.shape != (sample_count,) or categories.shape != (sample_count,):
        raise ValueError("sample_ids and event_type_ids must have one value per sample")
    if len(np.unique(sample_ids)) != sample_count:
        raise ValueError("sample_ids contain duplicates")
    valid = valid.astype(bool, copy=False)
    if not np.isfinite(predictions).all() or not np.isfinite(targets[valid]).all():
        raise ValueError("predictions and valid targets must be finite")
    if not np.isin(categories, list(EVENT_CATEGORIES)).all():
        raise ValueError("event_type_ids contain an unknown category code")

    return {
        "sample_ids": sample_ids,
        "predictions": predictions.astype(np.float64, copy=False),
        "targets": targets.astype(np.float64, copy=False),
        "valid_mask": valid,
        "event_mask": event.astype(bool, copy=False),
        "event_type_ids": categories.astype(np.int64, copy=False),
    }


def verify_prediction_pair(m0, m1):
    for name in ("sample_ids", "targets", "valid_mask", "event_mask", "event_type_ids"):
        if not np.array_equal(m0[name], m1[name]):
            raise ValueError(f"M0 and M1 differ in paired test array: {name}")
    if m0["predictions"].shape != m1["predictions"].shape:
        raise ValueError("M0 and M1 prediction shapes differ")


def masked_metrics(prediction, target, mask):
    count = int(mask.sum())
    if count == 0:
        return {
            "valid_targets": 0,
            "samples_with_valid_targets": 0,
            "mae": None,
            "rmse": None,
        }
    error = prediction[mask] - target[mask]
    return {
        "valid_targets": count,
        "samples_with_valid_targets": int(np.any(mask, axis=(1, 2)).sum()),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
    }


def paired_metrics(m0, m1, mask):
    left = masked_metrics(m0["predictions"], m0["targets"], mask)
    right = masked_metrics(m1["predictions"], m1["targets"], mask)
    if left["valid_targets"] != right["valid_targets"]:
        raise ValueError("M0 and M1 use different valid target counts")
    delta = {
        name: (None if left[name] is None else right[name] - left[name])
        for name in ("mae", "rmse")
    }
    return {"m0": left, "m1": right, "m1_minus_m0": delta}


def summarize_predictions(m0, m1):
    verify_prediction_pair(m0, m1)
    valid = m0["valid_mask"]
    event = m0["event_mask"][:, None, :]
    groups = {
        "all_nodes": valid,
        "event_nodes": valid & event,
        "other_nodes": valid & ~event,
    }
    result = {
        "sample_count": int(len(m0["sample_ids"])),
        "sample_id_min": int(m0["sample_ids"].min()),
        "sample_id_max": int(m0["sample_ids"].max()),
        "shape_samples_horizons_nodes": list(m0["predictions"].shape),
        "overall": {},
        "by_group_and_horizon": {},
        "by_category": {"event_nodes": {}, "all_nodes": {}},
        "by_category_and_horizon_event_nodes": {},
    }

    for group_name, mask in groups.items():
        result["overall"][group_name] = paired_metrics(m0, m1, mask)
        result["by_group_and_horizon"][group_name] = {}
        for step in range(mask.shape[1]):
            horizon_mask = np.zeros_like(mask)
            horizon_mask[:, step, :] = mask[:, step, :]
            result["by_group_and_horizon"][group_name][str(step + 1)] = paired_metrics(
                m0, m1, horizon_mask
            )

    for category_id, category_name in EVENT_CATEGORIES.items():
        category_samples = m0["event_type_ids"] == category_id
        event_mask = valid & event & category_samples[:, None, None]
        all_mask = valid & category_samples[:, None, None]
        result["by_category"]["event_nodes"][category_name] = paired_metrics(m0, m1, event_mask)
        result["by_category"]["all_nodes"][category_name] = paired_metrics(m0, m1, all_mask)
        result["by_category_and_horizon_event_nodes"][category_name] = {}
        for step in range(valid.shape[1]):
            horizon_mask = np.zeros_like(event_mask)
            horizon_mask[:, step, :] = event_mask[:, step, :]
            result["by_category_and_horizon_event_nodes"][category_name][str(step + 1)] = paired_metrics(
                m0, m1, horizon_mask
            )
    return result


def resolve_prediction_paths(m0, m1, args, parser):
    if bool(args.m0_predictions) != bool(args.m1_predictions):
        parser.error("--m0-predictions and --m1-predictions must be supplied together")
    if args.m0_predictions:
        paths = (args.m0_predictions, args.m1_predictions)
        for path in paths:
            if not path.is_file():
                parser.error(f"Prediction archive does not exist: {path}")
        return paths

    paths = (m0.get("test_predictions"), m1.get("test_predictions"))
    if not all(paths):
        return None
    paths = tuple(Path(path) for path in paths)
    exists = tuple(path.is_file() for path in paths)
    if exists == (True, True):
        return paths
    if exists[0] != exists[1]:
        raise FileNotFoundError(f"Only one paired prediction archive is available: {paths}")
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m0-summary", type=Path, required=True)
    parser.add_argument("--m1-summary", type=Path, required=True)
    parser.add_argument("--m0-predictions", type=Path)
    parser.add_argument("--m1-predictions", type=Path)
    args = parser.parse_args()

    m0, m1 = read_summary(args.m0_summary), read_summary(args.m1_summary)
    result = compare(m0, m1)
    paths = resolve_prediction_paths(m0, m1, args, parser)
    if paths:
        result["prediction_breakdown"] = summarize_predictions(
            load_predictions(paths[0]), load_predictions(paths[1])
        )
        result["prediction_archives"] = [str(path) for path in paths]
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
