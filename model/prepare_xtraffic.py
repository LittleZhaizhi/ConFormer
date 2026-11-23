"""Inspect the XTraffic release and assemble one configured ConFormer variant.

Run from the project root with python -m model.prepare_xtraffic.
"""

import argparse
from pathlib import Path

import yaml

from lib.xtraffic_data import EVENT_VOCAB_SIZES, prepare_xtraffic, save_json


def build_model(config, adjacency, device="cpu"):
    """Construct one model from its branch config without starting training."""
    import torch

    from model.ConFormer import ConFormer

    model_args = dict(config["model_args"])
    model_args["event_vocab_size"] = EVENT_VOCAB_SIZES[config["model"]]
    model_args["supports"] = [torch.as_tensor(adjacency, device=device)]
    return ConFormer(**model_args).to(device)


def inspect_model(model, sample, training_scale):
    """Measure the fixed graph's hops on one real input without updating weights."""
    import torch

    device = next(model.parameters()).device
    history = torch.as_tensor(sample["x"], device=device).unsqueeze(0)
    event_ids = torch.as_tensor(sample["event_ids"], device=device).unsqueeze(0)
    graph = model.graph_propagate
    hops = []

    def record_fixed_hop(_module, inputs, output):
        # GCN propagates over the fixed support before the learned support.
        if len(hops) < graph.order:
            if not torch.isfinite(output).all():
                raise ValueError(f"Fixed graph hop {len(hops) + 1} contains non-finite values")
            hops.append({
                "input_max_abs": inputs[0].abs().max().item(),
                "output_max_abs": output.abs().max().item(),
            })

    was_training = model.training
    handle = graph.nconv.register_forward_hook(record_fixed_hop)
    try:
        model.eval()
        with torch.inference_mode():
            prediction = model(history, event_ids=event_ids)
            prediction = prediction * training_scale["std"] + training_scale["mean"]
            if len(hops) != graph.order:
                raise ValueError("The model did not run every fixed graph hop")
            expected = (1, *sample["y_raw"].shape)
            if tuple(prediction.shape) != expected or not torch.isfinite(prediction).all():
                raise ValueError(f"Expected a finite original-scale prediction of shape {expected}")
            return {
                "sample_id": sample["sample_id"],
                "event_table": list(model.acc_embedding.weight.shape),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "fixed_graph_hops": hops,
                "prediction_shape": list(prediction.shape),
                "prediction_raw_range": [prediction.min().item(), prediction.max().item()],
            }
    finally:
        handle.remove()
        model.train(was_training)


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
    parser.add_argument(
        "--build-model", action="store_true",
        help="Construct the model and inspect fixed graph propagation in one forward pass",
    )
    parser.add_argument("--device", default="cpu", help="Device for the forward inspection")
    parser.add_argument("--seed", type=int, default=2024, help="Model inspection seed")
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
    if args.build_model:
        from lib.utils import seed_everything

        seed_everything(args.seed)
        model = build_model(config, adjacency, args.device)
        inspection = inspect_model(model, first, report["training_scale"])
        inspection["seed"] = args.seed
        inspection["graph"] = report["graph"]["model_support"]
        save_json(args.output_dir / f"model_inspection_{config['model']}.json", inspection)
        print("Event table:", tuple(inspection["event_table"]))
        print("Trainable parameters:", inspection["trainable_parameters"])
        print("Fixed graph hops:", inspection["fixed_graph_hops"])


if __name__ == "__main__":
    main()
