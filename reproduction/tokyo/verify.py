"""Check Tokyo attention axes or run one real training batch."""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib.utils import print_log, seed_everything
from model.tokyo import (
    DATA_DIR, SEED, build_model, device_for, load_config, load_tokyo,
    record_protocol, write_json,
)
from lib.utils import MaskedMAELoss
from model.tokyo import make_optimizer
from model.tky_dual_axis import DualAxisAttention


@torch.no_grad()
def check_axes(device):
    # Uniform scores make every expected response exactly the mean of the values.
    layer = DualAxisAttention(model_dim=4, condition_dim=1, num_heads=2).to(device)
    layer.query.weight.zero_()
    layer.key_value.weight.zero_()
    layer.key_value.weight[4:, :4].copy_(torch.eye(4, device=device))
    layer.key_value.weight[4:, -1] = 1

    x = torch.zeros(2, 3, 5, 4, device=device)
    condition = torch.zeros(2, 1, 5, 1, device=device)
    x[0, 1, 2] = 1
    qkv = layer.project(x, condition)
    expected_space = torch.zeros_like(x)
    expected_space[0, 1, :] = 1 / 5
    expected_time = torch.zeros_like(x)
    expected_time[0, :, 2] = 1 / 3
    torch.testing.assert_close(layer.attend_space(*qkv), expected_space)
    torch.testing.assert_close(layer.attend_time(*qkv), expected_time)

    # Changing only graph context must change K/V, while Q stays independent.
    layer.key_value.weight[:4, -1] = 1
    query, key, value = layer.project(x, condition)
    condition[0, 0, 2] = 1
    changed_query, changed_key, changed_value = layer.project(x, condition)
    delta = torch.zeros_like(key)
    delta[0, :, 2] = 1
    torch.testing.assert_close(changed_query, query)
    torch.testing.assert_close(changed_key - key, delta)
    torch.testing.assert_close(changed_value - value, delta)
    if layer(x, condition).shape != x.shape:
        raise RuntimeError("Dual-axis fusion changed the input shape")
    print("PASS axes: space over nodes, time over history, graph context in K/V")

def load_weights(model, path, device):
    path = path.resolve(strict=True)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))

def smoke(model, build_model, train_loader, val_loader, scaler, device, config, out, log):
    model.train()
    x, y = (tensor.to(device) for tensor in next(iter(train_loader)))
    optimizer = make_optimizer(model, config)
    before = model.output_proj.weight.detach().clone()
    predictions = scaler.inverse_transform(model(x))
    if predictions.shape != y.shape or not torch.isfinite(predictions).all():
        raise RuntimeError("Training predictions must be finite and match the target shape")
    if not torch.isfinite(y).all() or not y.ne(0).any():
        raise RuntimeError("Smoke batch requires finite targets and some nonzero truth")
    criterion = MaskedMAELoss()
    loss = criterion(predictions, y)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite training loss")
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(grad).all() for grad in gradients):
        raise RuntimeError("Missing or non-finite gradients")
    optimizer.step()
    if torch.equal(before, model.output_proj.weight.detach()):
        raise RuntimeError("The prediction head did not update")
    training_loss = loss.item()
    del x, y, predictions, loss, gradients, optimizer
    model.zero_grad(set_to_none=True)

    checkpoint = out / "smoke.pt"
    torch.save(model.state_dict(), checkpoint)
    reloaded = build_model()
    load_weights(reloaded, checkpoint, device)
    model.eval()
    reloaded.eval()
    with torch.inference_mode():
        x, y = (tensor.to(device) for tensor in next(iter(val_loader)))
        expected = scaler.inverse_transform(model(x))
        prediction = scaler.inverse_transform(reloaded(x))
        torch.testing.assert_close(prediction, expected)
        if prediction.shape != y.shape or not torch.isfinite(prediction).all():
            raise RuntimeError("Reloaded model produced invalid validation predictions")
        val_loss = criterion(prediction, y).item()
        if not np.isfinite(val_loss):
            raise RuntimeError("Non-finite validation loss")
    report = {
        "train_masked_mae": training_loss,
        "validation_masked_mae": val_loss,
        "prediction_shape": list(prediction.shape),
        "reloaded_checkpoint": str(checkpoint),
    }
    write_json(out / "smoke_report.json", report)
    print_log("PASS smoke: one update, finite predictions, matching reloaded output", log=log)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("check", choices=("axes", "smoke"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, help="New run directory for the smoke check")
    args = parser.parse_args()
    device = device_for(args.device)
    if args.check == "axes":
        check_axes(device)
        return
    if args.output_dir is None:
        parser.error("smoke requires --output-dir")

    config = load_config()
    seed_everything(SEED)

    def build():
        return build_model(args.data_dir, config, device)

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    with (out / "run.log").open("w", encoding="utf-8") as log:
        (train, val, _, scaler), _, data_files = load_tokyo(args.data_dir, config, log)
        model = build()
        record_protocol(out, "smoke", config, scaler, model, device, data_files, Path(__file__))
        smoke(model, build, train, val, scaler, device, config, out, log)


if __name__ == "__main__":
    main()
