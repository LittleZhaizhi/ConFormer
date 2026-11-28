"""Train the full Tokyo ConFormer and save its best validation checkpoint."""

import argparse
import datetime
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.utils import MaskedMAELoss, print_log, seed_everything
from model.tokyo import (
    DATA_DIR, SEED, build_model, device_for, load_config, load_tokyo, make_optimizer,
    record_protocol, write_json,
)


def train(model, train_loader, val_loader, scaler, config, device, checkpoint, log):
    criterion = MaskedMAELoss()
    optimizer = make_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=config["milestones"], gamma=config["lr_decay_rate"]
    )
    best = None
    wait = 0
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        train_losses = []
        for x, y in train_loader:
            prediction = scaler.inverse_transform(model(x.to(device)))
            loss = criterion(prediction, y.to(device))
            train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            if config.get("clip_grad"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad"])
            optimizer.step()
        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                prediction = scaler.inverse_transform(model(x.to(device)))
                val_losses.append(criterion(prediction, y.to(device)).item())
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        if not np.isfinite([train_loss, val_loss]).all():
            raise RuntimeError(f"Non-finite loss at epoch {epoch}")

        print_log(datetime.datetime.now(), f"Epoch {epoch}  Train Loss = {train_loss:.5f}"
                  f" Val Loss = {val_loss:.5f}", log=log)
        if best is None or val_loss < best["validation_loss"]:
            torch.save(model.state_dict(), checkpoint)
            best = {"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss}
            wait = 0
        else:
            wait += 1
            if wait >= config["early_stop"]:
                break

    print_log(f"Finished at epoch {epoch}; best epoch {best['epoch']}: "
              f"Val Loss = {best['validation_loss']:.5f}", log=log)
    print_log(f"Saved best weights: {checkpoint}", log=log)
    return {"completed_epochs": epoch, "best": best, "checkpoint": str(checkpoint)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, required=True, help="New training run directory")
    args = parser.parse_args()
    try:
        device = device_for(args.device)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))

    config = load_config()
    seed_everything(SEED)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    with (out / "run.log").open("w", encoding="utf-8") as log:
        print_log(f"Run: {out} | dual-axis | {device} | seed {SEED}", log=log)
        (train_loader, val_loader, _, scaler), _, data_files = load_tokyo(args.data_dir, config, log)
        model = build_model(args.data_dir, config, device)
        record_protocol(out, "train", config, scaler, model, device, data_files, Path(__file__))
        report = train(model, train_loader, val_loader, scaler, config, device, out / "best.pt", log)
        write_json(out / "train_report.json", report)


if __name__ == "__main__":
    main()
