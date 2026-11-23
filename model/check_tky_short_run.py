"""Run a small Tokyo train/validation/checkpoint/test check on the school server."""

import argparse
from itertools import islice
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
from lib.metrics import RMSE_MAE_MAPE
from lib.utils import MaskedMAELoss, print_log, seed_everything
from model.ConFormer import ConFormer
from model.train import load_tokyo_support, normalize_incoming_weights

BATCH_SIZE = 1
TRAIN_BATCHES = 8
VALIDATION_BATCHES = 2
TEST_BATCHES = 2


def predict_batch(model, x_batch, y_batch, scaler, device):
    targets = y_batch.to(device)
    predictions = scaler.inverse_transform(model(x_batch.to(device)))
    if predictions.shape != targets.shape:
        raise RuntimeError(f"Prediction shape {predictions.shape} differs from target shape {targets.shape}")
    if not torch.isfinite(predictions).all() or not torch.isfinite(targets).all():
        raise RuntimeError("The batch contains non-finite predictions or targets")
    return targets, predictions


@torch.no_grad()
def evaluate(model, loader, batch_limit, scaler, device, label, log):
    model.eval()
    targets = []
    predictions = []
    for x_batch, y_batch in islice(loader, batch_limit):
        y_true, y_pred = predict_batch(model, x_batch, y_batch, scaler, device)
        targets.append(y_true.cpu().numpy())
        predictions.append(y_pred.cpu().numpy())

    if not targets:
        raise RuntimeError(f"{label} has no batches")
    y_true = np.concatenate(targets, axis=0)
    y_pred = np.concatenate(predictions, axis=0)
    if not np.any(y_true != 0):
        raise RuntimeError(f"{label} has no unmasked targets")
    rmse, mae, mape = RMSE_MAE_MAPE(y_true, y_pred)
    if not np.isfinite([rmse, mae, mape]).all():
        raise RuntimeError(f"{label} produced a non-finite metric")
    print_log(
        f"{label} ({len(targets)} batches, {len(y_true)} windows): "
        f"RMSE={rmse:.6f}, MAE={mae:.6f}, MAPE={mape:.4f}%",
        log=log,
    )


def run_short_check(data_dir, model_args, config, device, run_dir, log, *, normalize_incoming=False):
    print_log("Loading the published Tokyo splits and constructing the model.", log=log)
    train_loader, val_loader, test_loader, scaler = get_dataloaders_from_index_data(
        data_dir,
        batch_size=BATCH_SIZE,
        in_steps=config["in_steps"],
        out_steps=config["out_steps"],
        tokyo_external=True,
        log=log,
    )
    support = load_tokyo_support(data_dir, model_args["num_nodes"])
    graph_mode = "published"
    if normalize_incoming:
        support = normalize_incoming_weights(support)
        graph_mode = "incoming-normalized"
    min_sum, max_sum = torch.aminmax(support.sum(dim=0))
    print_log(
        f"Fixed road graph: {graph_mode}. Column sum range: "
        f"[{min_sum.item():.6g}, {max_sum.item():.6g}]",
        log=log,
    )
    model_args["supports"] = [support]
    model = ConFormer(**model_args).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 0),
        eps=config.get("eps", 1e-8),
    )
    criterion = MaskedMAELoss()

    evaluate(model, val_loader, VALIDATION_BATCHES, scaler, device, "Validation before training", log)

    model.train()
    for step, (x_batch, y_batch) in enumerate(islice(train_loader, TRAIN_BATCHES), start=1):
        targets, predictions = predict_batch(model, x_batch, y_batch, scaler, device)
        valid = targets.ne(0)
        if not valid.any():
            raise RuntimeError(f"Training batch {step} has no unmasked targets")
        loss = criterion(predictions, targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Training batch {step} has non-finite loss")
        valid_predictions = predictions.detach()[valid]
        valid_targets = targets[valid]
        print_log(
            f"Train batch {step}/{TRAIN_BATCHES}: masked MAE={loss.item():.6f}, "
            f"prediction range=[{valid_predictions.min().item():.3f}, {valid_predictions.max().item():.3f}], "
            f"target range=[{valid_targets.min().item():.3f}, {valid_targets.max().item():.3f}]",
            log=log,
        )

        optimizer.zero_grad()
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f"Training batch {step} has non-finite gradient in {name}")
        if config.get("clip_grad"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad"])
        optimizer.step()

    evaluate(model, val_loader, VALIDATION_BATCHES, scaler, device, "Validation after training", log)

    checkpoint = run_dir / "short_run.pt"
    torch.save(model.state_dict(), checkpoint)
    print_log(f"Checkpoint saved: {checkpoint}", log=log)

    reloaded_model = ConFormer(**model_args)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    reloaded_model.load_state_dict(state)
    reloaded_model.to(device)
    print_log("Checkpoint loaded from disk into a fresh model.", log=log)
    evaluate(reloaded_model, test_loader, TEST_BATCHES, scaler, device, "Test subset after reload", log)
    print_log("PASS: Tokyo short run completed training, validation, and checkpoint reload.", log=log)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True, help="Visible CUDA device index")
    parser.add_argument(
        "--normalize-incoming",
        action="store_true",
        help="Normalize the fixed Tokyo graph by incoming weight sums",
    )
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
    run_dir = Path(mkdtemp(prefix="tky-short-", dir=logs_dir))
    log_path = run_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as log:
        print_log(f"Device: {device} ({torch.cuda.get_device_name(device)})", log=log)
        print_log(f"Log: {log_path}", log=log)
        run_short_check(
            data_dir,
            dict(config["model_args"]),
            config,
            device,
            run_dir,
            log,
            normalize_incoming=args.normalize_incoming,
        )


if __name__ == "__main__":
    main()
