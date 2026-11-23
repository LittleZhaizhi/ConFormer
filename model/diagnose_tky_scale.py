"""Trace prediction scales for the first Tokyo validation window on the server."""

import argparse
from functools import partial
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
from model.train import load_tokyo_support

GLN_FIELDS = (
    "attention shift",
    "attention scale",
    "attention gate",
    "feed-forward shift",
    "feed-forward scale",
    "feed-forward gate",
)


def trace_tensor(label, tensor, log):
    tensor = tensor.detach()
    finite_count = torch.isfinite(tensor).sum().item()
    if finite_count != tensor.numel():
        print_log(f"{label}: finite={finite_count}/{tensor.numel()}", log=log)
        raise RuntimeError(f"Non-finite values first observed at {label}")

    minimum, maximum = torch.aminmax(tensor)
    abs_p99 = torch.quantile(tensor.abs().float().flatten(), 0.99).item()
    print_log(
        f"{label}: shape={tuple(tensor.shape)}, "
        f"min={minimum.item():.6g}, max={maximum.item():.6g}, abs_p99={abs_p99:.6g}",
        log=log,
    )


def register_trace_hooks(model, fixed_support, trace):
    graph = model.graph_propagate
    hop_counts = {"published graph": 0, "learned graph": 0}

    def trace_hop(module, inputs, output):
        label = "published graph" if inputs[1] is fixed_support else "learned graph"
        hop_counts[label] += 1
        trace(f"{label}, hop {hop_counts[label]}", output)

    def trace_gln(prefix, module, inputs, output):
        for name, values in zip(GLN_FIELDS, output.chunk(len(GLN_FIELDS), dim=-1)):
            trace(f"{prefix} GLN {name}", values)

    def trace_layer(prefix, module, inputs, output):
        trace(f"{prefix} modulated input", inputs[0])
        trace(f"{prefix} output before gating", output)

    def trace_residual(prefix, branch_names, module, inputs, output):
        trace(f"{prefix} {next(branch_names)} residual contribution", output)

    handles = [
        graph.register_forward_pre_hook(lambda module, inputs: trace("Graph input", inputs[0])),
        graph.nconv.register_forward_hook(trace_hop),
        graph.register_forward_hook(lambda module, inputs, out: trace("Graph output to GLN", out)),
    ]
    for number, block in enumerate(model.attn_layers, start=1):
        prefix = f"Block {number}"
        handles.extend(
            (
                block.GLN.register_forward_hook(partial(trace_gln, prefix)),
                block.attn.register_forward_hook(partial(trace_layer, f"{prefix} attention")),
                block.feed_forward.register_forward_hook(partial(trace_layer, f"{prefix} feed-forward")),
                block.dropout.register_forward_hook(
                    partial(trace_residual, prefix, iter(("attention", "feed-forward")))
                ),
            )
        )
    return handles


@torch.no_grad()
def diagnose(config, data_dir, device, log):
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
    fixed_support = load_tokyo_support(data_dir, model_args["num_nodes"])
    model_args["supports"] = [fixed_support]
    model = ConFormer(**model_args).to(device).eval()

    with np.load(data_dir / "index.npz", allow_pickle=False) as splits:
        print_log(f"Validation window 0 [x_start, x_end, y_end]: {splits['val'][0].tolist()}", log=log)
    x_batch, targets = next(iter(val_loader))
    x_batch, targets = x_batch.to(device), targets.to(device)
    valid = targets.ne(0)
    if not valid.any():
        raise RuntimeError("The first validation window has no unmasked targets")
    print_log(f"Training scaler: mean={scaler.mean:.6g}, std={scaler.std:.6g}", log=log)

    trace = partial(trace_tensor, log=log)
    trace("Standardized traffic input", x_batch[..., 0])
    trace("Nonzero traffic targets", targets[valid])
    handles = register_trace_hooks(model, fixed_support, trace)
    try:
        raw_predictions = model(x_batch)
    finally:
        for handle in handles:
            handle.remove()

    if raw_predictions.shape != targets.shape:
        raise RuntimeError(f"Prediction shape {raw_predictions.shape} differs from {targets.shape}")
    trace("Model output before inverse scaling", raw_predictions)
    predictions = scaler.inverse_transform(raw_predictions)
    trace("Prediction after inverse scaling (unmasked targets)", predictions[valid])
    loss = MaskedMAELoss()(predictions, targets)
    print_log(f"Masked MAE: {loss.item():.6f}", log=log)
    print_log("DONE: Tokyo prediction-scale trace for validation window 0.", log=log)


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
    run_dir = Path(mkdtemp(prefix="tky-scale-", dir=logs_dir))
    log_path = run_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as log:
        print_log(f"Device: {device} ({torch.cuda.get_device_name(device)})", log=log)
        print_log(f"Log: {log_path}", log=log)
        print_log("Initial weights: seed 2024. Published graph weights. Eval mode.", log=log)
        diagnose(config, data_dir, device, log)


if __name__ == "__main__":
    main()
