"""Compare the scale of Tokyo predictions with two fixed road graph weightings."""

import argparse
from pathlib import Path
import sys
from tempfile import mkdtemp

import numpy as np
import torch
import yaml

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODEL_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from lib.data_prepare import get_dataloaders_from_index_data
from lib.utils import MaskedMAELoss, print_log, seed_everything
from model.ConFormer import ConFormer
from model.diagnose_tky_scale import register_trace_hooks, trace_tensor
from model.train import load_tokyo_support, normalize_incoming_weights

SUMMARY_TENSORS = (
    ("Graph input", "Graph input"),
    ("published graph, hop 1", "Fixed graph hop 1"),
    ("published graph, hop 2", "Fixed graph hop 2"),
    ("learned graph, hop 2", "Learned graph hop 2"),
    ("Graph output to GLN", "Graph output to GLN"),
    ("Block 1 GLN attention gate", "Block 1 attention gate"),
    ("Block 1 attention output before gating", "Block 1 attention before gating"),
    ("Block 1 attention residual contribution", "Block 1 attention after gating"),
    ("Block 1 feed-forward residual contribution", "Block 1 feed-forward after gating"),
    ("Model output before inverse scaling", "Model output before inverse scaling"),
    ("Prediction after inverse scaling (unmasked targets)", "Prediction on unmasked targets"),
)
SUMMARY_LABELS = dict(SUMMARY_TENSORS)


def run_forward(model, support, inputs, targets, valid, scaler, log):
    summary = {}

    def trace(label, tensor):
        trace_tensor(SUMMARY_LABELS.get(label, label), tensor, log)
        if label in SUMMARY_LABELS:
            summary[label] = torch.quantile(tensor.detach().abs().float().flatten(), 0.99).item()

    graph = model.graph_propagate
    original_supports = graph.supports
    handles = register_trace_hooks(model, support, trace)
    try:
        graph.supports = [support]
        raw_predictions = model(inputs)
    finally:
        graph.supports = original_supports
        for handle in handles:
            handle.remove()

    if raw_predictions.shape != targets.shape:
        raise RuntimeError(f"Prediction shape {raw_predictions.shape} differs from {targets.shape}")
    trace("Model output before inverse scaling", raw_predictions)
    predictions = scaler.inverse_transform(raw_predictions)
    trace("Prediction after inverse scaling (unmasked targets)", predictions[valid])
    masked_mae = MaskedMAELoss()(predictions, targets).item()
    return summary, masked_mae


@torch.no_grad()
def compare(config, data_dir, device, log):
    train_loader, val_loader, test_loader, scaler = get_dataloaders_from_index_data(
        data_dir,
        batch_size=1,
        in_steps=config["in_steps"],
        out_steps=config["out_steps"],
        tokyo_external=True,
        log=log,
    )
    del train_loader, test_loader

    model_args = dict(config["model_args"])
    published = load_tokyo_support(data_dir, model_args["num_nodes"])
    normalized = normalize_incoming_weights(published)
    model_args["supports"] = [published]
    model = ConFormer(**model_args).to(device).eval()

    with np.load(data_dir / "index.npz", allow_pickle=False) as splits:
        window = splits["val"][0].tolist()
    print_log(f"Validation window 0 [x_start, x_end, y_end]: {window}", log=log)
    print_log(f"Training scaler: mean={scaler.mean:.6g}, std={scaler.std:.6g}", log=log)

    inputs, targets = next(iter(val_loader))
    inputs, targets = inputs.to(device), targets.to(device)
    valid = targets.ne(0)
    if not valid.any():
        raise RuntimeError("The first validation window has no unmasked targets")
    trace_tensor("Nonzero traffic targets", targets[valid], log)

    results = []
    for case, support in (("Published", published), ("Incoming-normalized", normalized)):
        minimum, maximum = torch.aminmax(support.sum(dim=0))
        print_log(f"\n{case} fixed graph", log=log)
        print_log(f"Column sum range: [{minimum.item():.6g}, {maximum.item():.6g}]", log=log)
        results.append(run_forward(model, support, inputs, targets, valid, scaler, log))

    (published_stats, published_mae), (normalized_stats, normalized_mae) = results
    print_log("\nScale comparison on the same validation window", log=log)
    print_log(f"{'Measure':<47} {'Published':>12} {'Normalized':>12}", log=log)
    for key, label in SUMMARY_TENSORS:
        print_log(
            f"{label + ' abs_p99':<47} {published_stats[key]:>12.6g} {normalized_stats[key]:>12.6g}",
            log=log,
        )
    print_log(f"{'Masked MAE':<47} {published_mae:>12.6g} {normalized_mae:>12.6g}", log=log)
    print_log("DONE: Tokyo graph-scale comparison completed.", log=log)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True, help="Visible CUDA device index")
    args = parser.parse_args()
    gpu_count = torch.cuda.device_count()
    if not 0 <= args.gpu < gpu_count:
        raise RuntimeError(f"Requested CUDA device {args.gpu}, but {gpu_count} device(s) are visible")
    device = torch.device(f"cuda:{args.gpu}")
    seed_everything(2024)

    data_dir = PROJECT_DIR / "data" / "TKY"
    index_path = data_dir / "index.npz"
    if not index_path.is_file():
        raise FileNotFoundError(f"Published Tokyo split is required: {index_path}")
    with (MODEL_DIR / "ConFormer.yaml").open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)["TKY"]

    logs_dir = PROJECT_DIR / "logs"
    logs_dir.mkdir(exist_ok=True)
    log_path = Path(mkdtemp(prefix="tky-graph-compare-", dir=logs_dir)) / "run.log"
    with log_path.open("w", encoding="utf-8") as log:
        print_log(f"Device: {device} ({torch.cuda.get_device_name(device)})", log=log)
        print_log(f"Log: {log_path}", log=log)
        print_log("Same initial weights (seed 2024), validation window, and scaler. Eval mode.", log=log)
        compare(config, data_dir, device, log)


if __name__ == "__main__":
    main()
