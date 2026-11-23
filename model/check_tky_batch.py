"""Check a single Tokyo training batch on the school server."""

# Run with the remote SSH interpreter and a selected CUDA device.

import argparse
from pathlib import Path
import sys

import torch
import yaml

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODEL_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from lib.data_prepare import get_dataloaders_from_index_data
from lib.utils import MaskedMAELoss, seed_everything
from model.ConFormer import ConFormer
from model.train import load_tokyo_support


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True, help="Visible CUDA device index")
    args = parser.parse_args()

    gpu_count = torch.cuda.device_count()
    if not 0 <= args.gpu < gpu_count:
        raise RuntimeError(f"Requested CUDA device {args.gpu}, but {gpu_count} device(s) are visible")
    device = torch.device(f"cuda:{args.gpu}")
    seed_everything(2024)
    print(f"Device: {device} ({torch.cuda.get_device_name(device)})", flush=True)

    data_dir = PROJECT_DIR / "data" / "TKY"
    index_path = data_dir / "index.npz"
    if not index_path.is_file():
        raise FileNotFoundError(f"Published Tokyo split is required: {index_path}")
    with (MODEL_DIR / "ConFormer.yaml").open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)["TKY"]

    print("1. Constructing the model with the Tokyo road graph", flush=True)
    model_args = dict(config["model_args"])
    num_nodes = model_args["num_nodes"]
    support = load_tokyo_support(data_dir, num_nodes)
    model_args["supports"] = [support]
    model = ConFormer(**model_args).to(device)
    model.train()
    print(f"Road graph: {tuple(support.shape)}", flush=True)

    print("2. Loading one batch from the published training split", flush=True)
    train_loader, _, _, scaler = get_dataloaders_from_index_data(
        data_dir,
        in_steps=config["in_steps"],
        out_steps=config["out_steps"],
        batch_size=1,
        tokyo_external=True,
    )
    x_batch, y_batch = next(iter(train_loader))
    expected_input = (1, model_args["in_steps"], num_nodes, 5)
    expected_target = (1, model_args["out_steps"], num_nodes, model_args["output_dim"])
    if x_batch.shape != expected_input or y_batch.shape != expected_target:
        raise ValueError(
            f"Expected input {expected_input} and target {expected_target}, "
            f"got {tuple(x_batch.shape)} and {tuple(y_batch.shape)}"
        )
    if not torch.isfinite(x_batch).all() or not torch.isfinite(y_batch).all():
        raise ValueError("The training batch contains non-finite inputs or targets")
    if not y_batch.ne(0).any():
        raise ValueError("All targets in the selected batch are masked")
    time_of_day = x_batch[..., 1]
    day_of_week = x_batch[..., 2]
    events = x_batch[..., 3:5]
    if ((time_of_day < 0) | (time_of_day >= 1)).any():
        raise ValueError("Time-of-day values must be in [0, 1)")
    if ((day_of_week < 0) | (day_of_week > 6) | (day_of_week != day_of_week.floor())).any():
        raise ValueError("Day-of-week values must be integers from 0 to 6")
    if not ((events == 0) | (events == 1)).all():
        raise ValueError("The event channels must be binary")
    print(
        f"Input: {tuple(x_batch.shape)}, target: {tuple(y_batch.shape)}, "
        f"event values equal to 1 (channels 3, 4): {events.sum(dim=(0, 1, 2)).int().tolist()}",
        flush=True,
    )

    print("3. Running the forward pass and computing masked MAE", flush=True)
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)
    predictions = scaler.inverse_transform(model(x_batch))
    if predictions.shape != y_batch.shape or not torch.isfinite(predictions).all():
        raise RuntimeError("Predictions have the wrong shape or contain non-finite values")
    loss = MaskedMAELoss()(predictions, y_batch)
    if not torch.isfinite(loss):
        raise RuntimeError("The training loss is non-finite")
    print(f"Predictions: {tuple(predictions.shape)}, masked MAE: {loss.item():.6f}", flush=True)

    print("4. Checking gradients and one Adam update", flush=True)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 0),
        eps=config.get("eps", 1e-8),
    )
    optimizer.zero_grad()
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"Non-finite gradient in {name}")
    if config.get("clip_grad"):
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad"])
    weight = model.output_proj.weight
    before = weight.detach().clone()
    optimizer.step()
    if not torch.isfinite(weight).all() or torch.equal(weight.detach(), before):
        raise RuntimeError("The output projection weights did not update to finite values")
    print("PASS: Tokyo training batch completed one parameter update", flush=True)


if __name__ == "__main__":
    main()
