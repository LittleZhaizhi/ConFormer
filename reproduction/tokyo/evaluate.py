"""Evaluate a saved full Tokyo ConFormer checkpoint."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib.utils import print_log, seed_everything
from model.tokyo import (
    DATA_DIR, SEED, build_model, device_for, file_digest, load_config, load_tokyo,
    record_protocol, write_json,
)


PAPER_TOKYO = {
    10: {"mae": 5.77, "rmse": 9.08, "mape_pct": 19.69},
    30: {"mae": 6.40, "rmse": 10.23, "mape_pct": 23.16},
    60: {"mae": 6.73, "rmse": 10.87, "mape_pct": 25.28},
}


def metrics(truth, prediction, valid):
    count = int(valid.sum())
    if not count:
        return {"valid_positions": 0, "mae": None, "rmse": None, "mape_pct": None}
    selected_truth = truth[valid].astype(np.float64)
    error = prediction[valid].astype(np.float64) - selected_truth
    return {
        "valid_positions": count,
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "mape_pct": float(np.abs(error / selected_truth).mean() * 100),
    }


def compare_paper(per_step):
    rows = {}
    for minutes, paper in PAPER_TOKYO.items():
        ours = per_step[str(minutes)]
        deviation = {
            metric: 100 * (ours[metric] - paper[metric]) / paper[metric]
            for metric in ("mae", "rmse")
        }
        deviation["mape_pp"] = ours["mape_pct"] - paper["mape_pct"]
        rows[str(minutes)] = {
            "paper": paper,
            "model": ours,
            "deviation": deviation,
            "mae_within_5pct": abs(deviation["mae"]) <= 5,
            "rmse_within_5pct": abs(deviation["rmse"]) <= 5,
            "mape_within_5pp": abs(deviation["mape_pp"]) <= 5,
        }
    return {
        "primary_within_tolerance": all(
            row["mae_within_5pct"] and row["rmse_within_5pct"] for row in rows.values()
        ),
        "mape_within_tolerance": all(row["mape_within_5pp"] for row in rows.values()),
        "by_minutes": rows,
    }


@torch.inference_mode()
def evaluate(model, loader, scaler, device, split, out, checkpoint, log):
    model.eval()
    truth_batches, prediction_batches, event_batches = [], [], []
    for x, y in loader:
        events = x[..., 3:5].ne(0).any(dim=1)
        prediction = scaler.inverse_transform(model(x.to(device)))
        if prediction.shape != y.shape or not torch.isfinite(prediction).all():
            raise RuntimeError("Test predictions must be finite and match the target shape")
        truth_batches.append(y.squeeze(-1).numpy())
        prediction_batches.append(prediction.squeeze(-1).cpu().numpy())
        event_batches.append(events.numpy())

    truth = np.concatenate(truth_batches)
    prediction = np.concatenate(prediction_batches)
    events = np.concatenate(event_batches)
    if len(truth) != len(split["test"]) or not np.isfinite(truth).all():
        raise RuntimeError("Test targets must be finite and follow the released test index")
    valid = truth != 0
    accident, regulation = events[..., 0], events[..., 1]
    groups = {"no_event": ~(accident | regulation), "accident": accident, "regulation": regulation}
    overall = metrics(truth, prediction, valid)
    overall["window_node_pairs"] = int(accident.size)
    scores = {"all": overall}
    for name, mask in groups.items():
        result = metrics(truth, prediction, valid & mask[:, None, :])
        result["window_node_pairs"] = int(mask.sum())
        scores[name] = result
    per_step = {
        str((step + 1) * 10): metrics(truth[:, step], prediction[:, step], valid[:, step])
        for step in range(truth.shape[1])
    }
    if not all(item["valid_positions"] for item in per_step.values()):
        raise RuntimeError("Every Tokyo forecast horizon needs at least one nonzero target")
    comparison = compare_paper(per_step)
    np.savez_compressed(
        out / "test_predictions.npz",
        window_index=split["test"],
        targets=truth,
        predictions=prediction,
        valid_mask=valid,
        accident_mask=accident,
        regulation_mask=regulation,
    )
    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_digest(checkpoint),
        "scores": scores,
        "accident_regulation_overlap": int((accident & regulation).sum()),
        "per_horizon_minutes": per_step,
        "paper_table_2": comparison,
    }
    write_json(out / "test_report.json", report)
    for minutes, row in comparison["by_minutes"].items():
        print_log(f"Table 2 {minutes} min:", row, log=log)
    for name, values in scores.items():
        print_log(f"All steps ({name}):", values, log=log)
    status = "PASS" if comparison["primary_within_tolerance"] else "REVIEW"
    print_log(f"{status} Table 2 MAE/RMSE tolerance; MAPE assessed separately", log=log)
    return report


def compare_reference(out, reference, report):
    """Compare the exact evaluation positions and the stored checkpoint predictions."""
    reference_report = json.loads((reference / "test_report.json").read_text(encoding="utf-8"))
    if report["checkpoint_sha256"] != reference_report["checkpoint_sha256"]:
        raise ValueError("Reference checkpoint fingerprint differs")

    with (
        np.load(reference / "test_predictions.npz", allow_pickle=False) as old,
        np.load(out / "test_predictions.npz", allow_pickle=False) as new,
    ):
        for key in ("window_index", "targets", "valid_mask", "accident_mask", "regulation_mask"):
            if not np.array_equal(new[key], old[key]):
                raise ValueError(f"Reference evaluation positions differ: {key}")
        expected, actual = old["predictions"], new["predictions"]
        if expected.shape != actual.shape:
            raise ValueError("Reference prediction shape differs")
        max_error = 0.0
        for start in range(0, len(expected), 32):
            before, after = expected[start:start + 32], actual[start:start + 32]
            if not np.allclose(after, before, rtol=1e-5, atol=1e-5):
                raise ValueError(f"Reference predictions differ in the batch starting at {start}")
            max_error = max(max_error, float(np.max(np.abs(after - before))))

    for item in ("scores", "per_horizon_minutes"):
        for name, expected in reference_report[item].items():
            actual = report[item][name]
            for key, value in expected.items():
                if key in ("mae", "rmse", "mape_pct") and value is not None:
                    equal = actual[key] is not None and np.isclose(actual[key], value, rtol=1e-5, atol=1e-5)
                else:
                    equal = actual[key] == value
                if not equal:
                    raise ValueError(f"Reference metric differs: {item}.{name}.{key}")
    if report["accident_regulation_overlap"] != reference_report["accident_regulation_overlap"]:
        raise ValueError("Reference event overlap differs")
    return {"reference_dir": str(reference), "passed": True, "max_prediction_abs_diff": max_error}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New evaluation directory")
    parser.add_argument("--reference-dir", type=Path, help="Optional completed evaluation to compare")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = device_for(args.device)
    checkpoint = args.checkpoint.resolve(strict=True)
    reference = args.reference_dir.resolve(strict=True) if args.reference_dir else None
    if reference:
        for name in ("test_report.json", "test_predictions.npz"):
            if not (reference / name).is_file():
                parser.error(f"Missing reference artifact: {reference / name}")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = load_config()
    seed_everything(SEED)
    with (out / "run.log").open("w", encoding="utf-8") as log:
        (_, _, test, scaler), split, data_files = load_tokyo(args.data_dir, config, log)
        model = build_model(args.data_dir, config, device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        record_protocol(out, "test", config, scaler, model, device, data_files, Path(__file__))
        report = evaluate(model, test, scaler, device, split, out, checkpoint, log)
        if reference:
            comparison = compare_reference(out, reference, report)
            write_json(out / "reference_check.json", comparison)
            print_log("PASS reference checkpoint, predictions, and metrics", comparison, log=log)


if __name__ == "__main__":
    main()
