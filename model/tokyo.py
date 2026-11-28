"""Data and model setup shared by Tokyo training and evaluation."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "model" / "ConFormer.yaml"
DATA_DIR = ROOT / "data" / "TKY"
SEED = 2024
SPLIT_COUNTS = {"train": 6557, "val": 2186, "test": 4493}


def load_config():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["TKY"]


def load_tokyo(data_dir, config, log):
    from lib.data_prepare import get_dataloaders_from_index_data

    data_dir = data_dir.resolve(strict=True)
    traffic = data_dir / "data.npz"
    if not traffic.is_file():
        traffic = data_dir / "data.h5"
    data_files = [traffic, *(data_dir / name for name in ("external.npz", "adj.npy", "index.npz"))]
    for path in data_files:
        if not path.is_file():
            raise FileNotFoundError(f"Missing Tokyo release file: {path}")

    with np.load(data_dir / "index.npz", allow_pickle=False) as index:
        split = {name: index[name] for name in SPLIT_COUNTS}
    for name, windows in split.items():
        if (
            windows.shape != (SPLIT_COUNTS[name], 3)
            or not np.issubdtype(windows.dtype, np.integer)
            or np.any(windows < 0)
            or not np.all(np.diff(windows, axis=1) == [config["in_steps"], config["out_steps"]])
        ):
            raise ValueError(f"Unexpected released Tokyo {name} split")

    loaders = get_dataloaders_from_index_data(
        data_dir, batch_size=config["batch_size"], tokyo_external=True,
        in_steps=config["in_steps"], out_steps=config["out_steps"], log=log,
    )
    return loaders, split, data_files


def build_model(data_dir, config, device):
    from model.ConFormer import ConFormer

    model_args = config["model_args"]
    nodes = model_args["num_nodes"]
    adjacency = np.load(data_dir / "adj.npy", allow_pickle=False)
    if adjacency.shape == (2841, 2841):
        index_path = ROOT / "model" / "tokyo_link_idx.csv"
        indices = np.loadtxt(index_path, dtype=np.intp, ndmin=1)
        if (
            indices.shape != (nodes,) or np.unique(indices).size != nodes
            or np.any(indices < 0) or np.any(indices >= adjacency.shape[0])
        ):
            raise ValueError(f"Invalid Tokyo road indices in {index_path}")
        adjacency = adjacency[np.ix_(indices, indices)]
    if adjacency.shape != (nodes, nodes) or not np.isfinite(adjacency).all():
        raise ValueError("Invalid Tokyo road adjacency")

    support = torch.from_numpy(np.asarray(adjacency, dtype=np.float32))
    # The graph convolution sums incoming edges along each receiver column.
    column_sums = support.sum(dim=0, keepdim=True)
    if not torch.isfinite(column_sums).all() or (column_sums <= 0).any():
        raise ValueError("Incoming normalization requires finite positive column sums")
    return ConFormer(**model_args, supports=[support / column_sums]).to(device)


def make_optimizer(model, config):
    return torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"],
        eps=config.get("eps", 1e-8),
    )


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def record_protocol(out, mode, config, scaler, model, device, data_files, entry):
    sources = [
        CONFIG, ROOT / "model/ConFormer.py", ROOT / "model/tky_dual_axis.py",
        ROOT / "model/tokyo.py", ROOT / "model/tokyo_link_idx.csv",
        ROOT / "lib/data_prepare.py", ROOT / "lib/utils.py", entry.resolve(),
    ]
    write_json(out / "protocol.json", {
        "architecture": "dual-axis",
        "mode": mode,
        "seed": SEED,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "normalize_incoming": True,
        "config": config,
        "split_counts": SPLIT_COUNTS,
        "training_scaler": {"mean": float(scaler.mean), "std": float(scaler.std)},
        "sources_sha256": {path.relative_to(ROOT).as_posix(): file_digest(path) for path in sources},
        "data_sha256": {str(path): file_digest(path) for path in data_files},
    })


def device_for(name):
    device = torch.device(name)
    if device.type == "cuda" and (
        not torch.cuda.is_available()
        or (device.index is not None and device.index >= torch.cuda.device_count())
    ):
        raise ValueError(f"CUDA device {device} is not available")
    return device
