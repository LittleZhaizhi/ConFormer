"""Train this branch on Alameda and archive its test predictions."""

import argparse
import hashlib
from math import sqrt
from pathlib import Path
import sys
from tempfile import mkdtemp
from time import perf_counter

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.utils import print_log, seed_everything
from lib.xtraffic_data import MODEL_VARIANT, prepare_xtraffic, save_json
from model.prepare_xtraffic import build_model, make_dataloaders


LEARNING_RATE = 0.001
EVENT_WEIGHT = "acc_embedding.weight"


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def backbone_state(model):
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if name != EVENT_WEIGHT
    }


def backbone_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.contiguous().numpy()
        digest.update(repr((name, array.shape, array.dtype.str)).encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def archive_initial_backbone(model, initial_source, run_dir):
    """Load an optional compatible initialization, then archive this run's own copy."""
    backbone = backbone_state(model)
    if initial_source is not None:
        initial = torch.load(initial_source, map_location="cpu", weights_only=True)
        if not isinstance(initial, dict) or initial.keys() != backbone.keys():
            raise ValueError("Expected an initial backbone file with only the shared tensors")
        for name, tensor in initial.items():
            expected = backbone[name]
            if (not isinstance(tensor, torch.Tensor)
                    or tensor.shape != expected.shape or tensor.dtype != expected.dtype):
                raise ValueError(f"Incompatible initial backbone tensor: {name}")
        event_weight = model.state_dict()[EVENT_WEIGHT]
        model.load_state_dict({**initial, EVENT_WEIGHT: event_weight}, strict=True)
        backbone = backbone_state(model)

    initial_path = run_dir / "initial_backbone.pt"
    torch.save(backbone, initial_path)
    return backbone_digest(backbone), initial_path


class ErrorTotals:
    """Aggregate MAE and RMSE over valid points, including across partial batches."""

    def __init__(self):
        self.count = 0
        self.absolute = 0.0
        self.squared = 0.0

    def add(self, errors):
        errors = errors.detach().double()
        self.count += errors.numel()
        self.absolute += errors.abs().sum().item()
        self.squared += errors.square().sum().item()

    def metrics(self):
        if self.count == 0:
            raise ValueError("The split has no valid targets")
        return {
            "valid_targets": self.count,
            "mae": self.absolute / self.count,
            "rmse": sqrt(self.squared / self.count),
        }


def predict_batch(model, batch, scale, device):
    targets = batch["y_raw"].to(device)
    valid = batch["valid_mask"].to(device)
    prediction = model(batch["x"].to(device), event_ids=batch["event_ids"].to(device))
    prediction = prediction * scale["std"] + scale["mean"]
    if prediction.shape != targets.shape or not torch.isfinite(prediction).all():
        raise ValueError(f"Expected finite predictions of shape {tuple(targets.shape)}")
    return prediction, targets, valid


def train_epoch(model, loader, optimizer, scale, device):
    model.train()
    totals = ErrorTotals()
    sample_order = hashlib.sha256()
    for batch in loader:
        sample_order.update(batch["sample_id"].numpy().astype("<i8", copy=False).tobytes())
        optimizer.zero_grad(set_to_none=True)
        prediction, targets, valid = predict_batch(model, batch, scale, device)
        errors = prediction[valid] - targets[valid]
        if errors.numel() == 0:
            continue
        loss = errors.abs().mean()
        if not torch.isfinite(loss):
            raise ValueError("Training produced a non-finite loss")
        loss.backward()
        optimizer.step()
        totals.add(errors)
    return totals.metrics(), sample_order.hexdigest()


@torch.inference_mode()
def evaluate(model, loader, scale, device, *, archive_path=None, event_type_ids=None):
    model.eval()
    totals = ErrorTotals()
    arrays = {
        "sample_ids": [], "predictions": [], "targets": [], "valid_mask": [],
        "event_mask": [], "event_type_ids": [],
    }
    for batch in loader:
        prediction, targets, valid = predict_batch(model, batch, scale, device)
        totals.add(prediction[valid] - targets[valid])
        if archive_path is not None:
            sample_ids = batch["sample_id"].numpy()
            arrays["sample_ids"].append(sample_ids)
            arrays["predictions"].append(prediction.cpu().numpy())
            arrays["targets"].append(batch["y_raw"].numpy())
            arrays["valid_mask"].append(batch["valid_mask"].numpy())
            arrays["event_mask"].append(batch["event_mask"].numpy())
            arrays["event_type_ids"].append(event_type_ids[sample_ids])

    metrics = totals.metrics()
    if archive_path is not None:
        np.savez_compressed(archive_path, **{
            name: np.concatenate(parts, axis=0) for name, parts in arrays.items()
        })
    return metrics


def train_to_best(model, loaders, scale, device, args, run_dir, log):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    checkpoint = run_dir / "best.pt"
    epochs = []
    best = None
    for epoch in range(1, args.max_epochs + 1):
        training, order_digest = train_epoch(model, loaders["train"], optimizer, scale, device)
        validation = evaluate(model, loaders["val"], scale, device)
        record = {
            "epoch": epoch,
            "train_order_sha256": order_digest,
            "training": training,
            "validation": validation,
        }
        epochs.append(record)
        print_log(f"Epoch {epoch:3d} | train MAE {training['mae']:.6f} | "
                  f"val MAE {validation['mae']:.6f} | val RMSE {validation['rmse']:.6f} | "
                  f"order {order_digest}", log=log)
        if best is None or validation["mae"] < best["validation"]["mae"]:
            best = record
            torch.save(model.state_dict(), checkpoint)
        if epoch - best["epoch"] >= args.patience:
            print_log(f"Early stopping at epoch {epoch}; best epoch {best['epoch']}.", log=log)
            break
    return checkpoint, best, epochs


def run(config, args, device, run_dir, log):
    dataset, splits, adjacency, audit = prepare_xtraffic(args.data_dir, run_dir, config)
    scale = audit["training_scale"]
    graph = audit["graph"]["model_support"]
    protocol = {
        "seed": args.seed,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "loss": "original-scale MAE at valid targets",
        "selection": "all-region validation MAE",
        "data_config": {key: value for key, value in config.items() if key != "model"},
        "split_manifest_sha256": file_digest(run_dir / "split_manifest.json"),
        "type_mapping_sha256": file_digest(args.data_dir / "type_mapping.json"),
        "graph_sha256": graph["sha256"],
        "training_scale": scale,
        "model_source_sha256": file_digest(ROOT / "model" / "ConFormer.py"),
        "loader_source_sha256": file_digest(ROOT / "lib" / "xtraffic_data.py"),
        "training_source_sha256": file_digest(Path(__file__)),
        "torch_version": str(torch.__version__),
    }
    save_json(run_dir / "protocol.json", protocol)
    print_log(f"Fixed graph columns {graph['column_sum_range']}, zero columns {graph['zero_columns']}, "
              f"SHA-256 {graph['sha256']}", log=log)

    seed_everything(args.seed)
    model = build_model(
        config, adjacency, device,
        isolate_event_init=args.initial_backbone is None,
    )
    initial_hash, initial_path = archive_initial_backbone(model, args.initial_backbone, run_dir)
    print_log(f"Initial backbone SHA-256: {initial_hash} ({initial_path})", log=log)
    print_log(f"Event table: {list(model.acc_embedding.weight.shape)}", log=log)

    # Start the training RNG at the configured seed after model initialization.
    seed_everything(args.seed)
    loaders = make_dataloaders(dataset, splits, args.batch_size, args.seed)
    start = perf_counter()
    checkpoint, best, epochs = train_to_best(
        model, loaders, scale, device, args, run_dir, log,
    )
    training_seconds = perf_counter() - start

    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    archive = run_dir / "test_predictions.npz"
    test = evaluate(model, loaders["test"], scale, device, archive_path=archive,
                    event_type_ids=dataset.event_types)
    print_log(f"Best epoch {best['epoch']} | test MAE {test['mae']:.6f} | "
              f"test RMSE {test['rmse']:.6f}", log=log)

    summary = run_dir / "run_summary.json"
    save_json(summary, {
        "model": config["model"],
        "device": str(device),
        "data_dir": str(args.data_dir),
        "protocol": protocol,
        "initial_backbone_sha256": initial_hash,
        "initial_backbone": str(initial_path),
        "initial_backbone_source": (
            str(args.initial_backbone) if args.initial_backbone is not None else None
        ),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "training_seconds": training_seconds,
        "epochs": epochs,
        "best": best,
        "test": test,
        "checkpoint": str(checkpoint),
        "test_predictions": str(archive),
    })
    print_log(f"PASS: {config['model'].upper()} seed {args.seed}. Report: {summary}", log=log)


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expect-model", choices=(MODEL_VARIANT,), required=True)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--batch-size", type=positive_int, default=16)
    parser.add_argument("--max-epochs", type=positive_int, default=100)
    parser.add_argument("--patience", type=positive_int, default=15)
    parser.add_argument(
        "--initial-backbone", type=Path,
        help="Optional shared, pre-training initial_backbone.pt for an exact initialization replay",
    )
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / "model" / "XTraffic.yaml").read_text(encoding="utf-8"))
    if config["model"] != args.expect_model:
        parser.error(f"XTraffic.yaml selects {config['model']}, expected {args.expect_model}")
    if not config["road_graph"]["normalize_incoming"]:
        parser.error("Alameda training requires the verified incoming-normalized fixed graph")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 4294967295")
    if args.initial_backbone is not None:
        args.initial_backbone = args.initial_backbone.resolve()
        if not args.initial_backbone.is_file():
            parser.error("--initial-backbone must point to an existing initialization file")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available in this Python environment")
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    run_dir = Path(mkdtemp(prefix=f"alameda-formal-{config['model']}-{args.seed}-", dir=logs))
    with (run_dir / "run.log").open("w", encoding="utf-8") as log:
        print_log(f"{config['model'].upper()} seed {args.seed} on {device}. Run: {run_dir}", log=log)
        run(config, args, device, run_dir, log)


if __name__ == "__main__":
    main()
