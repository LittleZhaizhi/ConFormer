"""Inspect the XTraffic release and assemble one configured ConFormer variant.

Run from the project root with python -m model.prepare_xtraffic.
"""

import argparse
from pathlib import Path

import yaml

from lib.xtraffic_data import EVENT_VOCAB_SIZE, prepare_xtraffic


def build_model(config, adjacency, device="cpu", *, isolate_event_init=True):
    """Construct one model from its branch config without starting training."""
    import torch

    from model.ConFormer import ConFormer

    model_args = dict(config["model_args"])
    model_args["event_vocab_size"] = EVENT_VOCAB_SIZE
    model_args["isolate_event_init"] = isolate_event_init
    model_args["supports"] = [torch.as_tensor(adjacency, device=device)]
    return ConFormer(**model_args).to(device)


def make_dataloaders(dataset, splits, batch_size, seed):
    """Give both variants the same train shuffle for a given seed."""
    import torch
    from torch.utils.data import DataLoader, Subset

    generator = torch.Generator().manual_seed(seed)
    return {
        name: DataLoader(
            Subset(dataset, indices),
            batch_size=batch_size,
            shuffle=name == "train",
            generator=generator if name == "train" else None,
        )
        for name, indices in splits.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", required=True, type=Path,
        help="Trusted IGSTGNN Alameda directory containing incident_all.npy",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="Directory for the manifest and audit",
    )
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("XTraffic.yaml"),
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dataset, splits, adjacency, report = prepare_xtraffic(
        args.data_dir, args.output_dir, config
    )
    first = dataset[0]
    print(f"{config['region']} / {config['model']}: {len(dataset)} samples")
    print("Splits:", {name: len(indices) for name, indices in splits.items()})
    print(f"History: {first['x'].shape}, target: {first['y_raw'].shape}")
    print(f"First sample selected nodes: {first['event_mask'].sum()}/{adjacency.shape[0]}")
    print("Candidate masks with no/all nodes:", report["all_zero_masks"], report["all_one_masks"])
    print("Fixed support:", adjacency.shape, report["graph"]["model_support"])
    print(f"Saved audit and split manifest in {args.output_dir}")


if __name__ == "__main__":
    main()
