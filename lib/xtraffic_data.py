"""Prepare IGSTGNN's released Alameda windows for the two ConFormer variants."""

import csv
import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np


EVENT_TYPES = ("hazard", "other", "accident", "breakdown", "police", "weather")
EVENT_TYPE_IDS = {name: index for index, name in enumerate(EVENT_TYPES, start=1)}
MODEL_VARIANT = "m0"
EVENT_VOCAB_SIZE = 2
CALENDAR_ATOL = 1e-4


def save_json(path, payload):
    """Write a report atomically so a partial write cannot be read as a result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.stem}-",
        suffix=".tmp", delete=False,
    ) as temporary:
        json.dump(payload, temporary, indent=2)
        temporary.write("\n")
    Path(temporary.name).replace(path)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(array):
    return hashlib.blake2b(array.tobytes(), digest_size=16).digest()


def released_splits(sample_count):
    """Use the IGSTGNN release order and its floor rounding for train and val."""
    train_end = int(sample_count * 0.70)
    val_end = train_end + int(sample_count * 0.15)
    if not 0 < train_end < val_end < sample_count:
        raise ValueError("The release must have a nonempty train, val and test split")
    return {
        "train": range(train_end),
        "val": range(train_end, val_end),
        "test": range(val_end, sample_count),
    }


def _read_type_names(path):
    mapping = json.loads(path.read_text(encoding="utf-8-sig"))
    if (
        not isinstance(mapping, dict)
        or set(mapping) != set(EVENT_TYPES)
        or any(type(code) is not int for code in mapping.values())
        or len(set(mapping.values())) != len(EVENT_TYPES)
    ):
        raise ValueError(f"Expected six distinct integer category codes in {path}")
    return {code: name for name, code in mapping.items()}


def _read_release_scale(path):
    with np.load(path, allow_pickle=False) as stats:
        normalized = stats["normalized"].item()
        mean = float(stats["mean"].item())
        std = float(stats["std"].item())
    if not isinstance(normalized, bool) or not np.isfinite([mean, std]).all() or std <= 0:
        raise ValueError(f"Invalid release normalization metadata in {path}")
    return normalized, mean, std


def _check_array(sample, key, expected_shape):
    array = np.asarray(sample[key])
    if array.shape != expected_shape or array.dtype.kind not in "fiu":
        raise ValueError(f"{key} must be a numeric array of shape {expected_shape}")
    return array


def _integer_field(value, name):
    scalar = np.asarray(value)
    if scalar.shape != () or scalar.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a single integer")
    value = scalar.item()
    if not np.isfinite(value) or value != int(value):
        raise ValueError(f"{name} must be a single integer")
    return int(value)


def _calendar_codes(history, steps_per_day):
    calendar = history[..., 1:].astype(np.float32)
    indices = calendar * np.array([steps_per_day, 7], dtype=np.float32)
    rounded = np.rint(indices)
    if not (
        np.all(np.isclose(indices, rounded, rtol=0, atol=CALENDAR_ATOL))
        and np.all((rounded >= 0) & (rounded < [steps_per_day, 7]))
    ):
        raise ValueError("History calendar values must align to the daily and weekly grids")
    # ConFormer preserves the TOD fraction and truncates after multiplying it.
    if not np.array_equal(indices[-1, :, 0].astype(np.int64), rounded[-1, :, 0]):
        raise ValueError("ConFormer would look up the wrong time-of-day slot at the last step")
    return rounded[..., 1]


def _sample_fields(sample, *, in_steps, out_steps, num_nodes, steps_per_day, type_names):
    if not isinstance(sample, dict):
        raise ValueError("Expected a single incident sample dictionary")
    history = _check_array(sample, "x_data", (in_steps, num_nodes, 3))
    future = _check_array(sample, "y_data", (out_steps, num_nodes, 3))
    relations = _check_array(sample, "incident_distances", (num_nodes, 3))
    if not np.isfinite(relations).all():
        raise ValueError("incident_distances contains a non-finite relation")
    event = sample["incident_features"]
    if not isinstance(event, dict):
        raise ValueError("incident_features must describe a single incident")
    type_code = _integer_field(event["Type"], "Type")
    if type_code not in type_names:
        raise ValueError(f"Unknown release Type code {type_code}")
    position = _integer_field(sample["incident_position"], "incident_position")
    _calendar_codes(history, steps_per_day)
    # Equivalent to IGSTGNN's abs().sum(-1) > 0 for finite relation values.
    mask = np.any(relations != 0, axis=-1)
    return history, future, mask, EVENT_TYPE_IDS[type_names[type_code]], position


def _restore_traffic(values, normalized, mean, std, missing_value):
    traffic = np.asarray(values, dtype=np.float64)
    if normalized:
        traffic = traffic * std + mean
    valid = np.isfinite(traffic)
    if missing_value is not None:
        # Compare on the source scale to avoid float round-off after inversion.
        sentinel = (missing_value - mean) / std if normalized else missing_value
        valid &= values != np.asarray(sentinel, dtype=values.dtype)
    return traffic, valid


class _TrafficMoments:
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.squared_deviation = 0.0

    def update(self, values):
        if not values.size:
            return
        mean = float(values.mean())
        total_count = self.count + values.size
        delta = mean - self.mean
        self.squared_deviation += float(np.square(values - mean).sum())
        self.squared_deviation += delta**2 * self.count * values.size / total_count
        self.mean += delta * values.size / total_count
        self.count = total_count

    def statistics(self):
        if not self.count:
            raise ValueError("No valid historical traffic values in the training split")
        std = float(np.sqrt(self.squared_deviation / self.count))
        if not np.isfinite([self.mean, std]).all() or std <= 0:
            raise ValueError("Training traffic must have a finite, positive standard deviation")
        return {"mean": self.mean, "std": std, "valid_count": self.count}


class XTrafficDataset:
    """Read one prepared window at a time without making a second copy of the release."""

    def __init__(
        self, samples, event_masks, event_types, steps_per_day,
        release_scale, targets_normalized, training_scale, missing_value,
    ):
        self.samples = samples
        self.event_masks = event_masks
        # Keep release labels for evaluation groups; the model receives 0 or 1.
        self.event_types = event_types
        self.steps_per_day = steps_per_day
        self.release_scale = release_scale
        self.targets_normalized = targets_normalized
        self.training_scale = training_scale
        self.missing_value = missing_value

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, sample_id):
        sample = self.samples[sample_id]
        x = np.array(sample["x_data"], dtype=np.float32, copy=True)
        history, history_valid = _restore_traffic(
            np.asarray(sample["x_data"])[..., 0], *self.release_scale, self.missing_value
        )
        mean = self.training_scale["mean"]
        std = self.training_scale["std"]
        # A missing history value maps to the training mean in model space.
        x[..., 0] = np.where(history_valid, (history - mean) / std, 0)
        x[..., 2] = _calendar_codes(x, self.steps_per_day)
        target, target_valid = _restore_traffic(
            np.asarray(sample["y_data"])[..., :1], self.targets_normalized,
            *self.release_scale[1:], self.missing_value,
        )
        mask = self.event_masks[sample_id]
        code = 1
        return {
            "sample_id": int(sample_id),
            "x": x,
            "y_raw": np.where(target_valid, target, 0).astype(np.float32),
            "valid_mask": target_valid,
            "event_mask": mask.copy(),
            "event_ids": np.where(mask, code, 0).astype(np.int64),
        }


def _graph_summary(matrix):
    row_sums = matrix.sum(axis=1, dtype=np.float64)
    column_sums = matrix.sum(axis=0, dtype=np.float64)
    return {
        "weight_range": [float(matrix.min()), float(matrix.max())],
        "nonzero_weights": int(np.count_nonzero(matrix)),
        "row_sum_range": [float(row_sums.min()), float(row_sums.max())],
        "column_sum_range": [float(column_sums.min()), float(column_sums.max())],
        "zero_columns": int(np.count_nonzero(column_sums == 0)),
        "sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
    }


def read_road_graph(data_dir, num_nodes, *, transpose=False, normalize_incoming=False):
    """Load one fixed support using ConFormer's source-row, destination-column convention."""
    with (data_dir / "sensors.csv").open(encoding="utf-8-sig", newline="") as file:
        station_ids = [row["station_id"].strip() for row in csv.DictReader(file)]
    if len(station_ids) != num_nodes or "" in station_ids or len(set(station_ids)) != num_nodes:
        raise ValueError(f"sensors.csv must contain {num_nodes} distinct station IDs")

    adjacency = np.load(data_dir / "adj_matrix.npy", allow_pickle=False)
    if adjacency.shape != (num_nodes, num_nodes) or adjacency.dtype.kind not in "fiu":
        raise ValueError(f"adj_matrix.npy must be a numeric {num_nodes} x {num_nodes} matrix")
    if not np.isfinite(adjacency).all() or (adjacency < 0).any():
        raise ValueError("Fixed road graph weights must be finite and nonnegative")
    original = _graph_summary(adjacency)
    if transpose:
        adjacency = adjacency.T
    if normalize_incoming:
        column_sums = adjacency.sum(axis=0, dtype=np.float64)
        if not np.isfinite(column_sums).all():
            raise ValueError("Incoming normalization requires finite column sums")
        # Nodes with no incoming edges keep their zero columns.
        adjacency = np.divide(
            adjacency,
            column_sums,
            out=np.zeros_like(adjacency, dtype=np.float64),
            where=column_sums > 0,
        )

    adjacency = np.ascontiguousarray(adjacency, dtype=np.float32)
    if not np.isfinite(adjacency).all():
        raise ValueError("Fixed road graph weights exceed the model's float32 range")
    first_hop = np.ones(num_nodes, dtype=np.float32) @ adjacency
    second_hop = first_hop @ adjacency
    if not np.isfinite(first_hop).all() or not np.isfinite(second_hop).all():
        raise ValueError("Fixed graph propagation overflows in float32")
    return adjacency, {
        "release": original,
        "model_support": _graph_summary(adjacency),
        "transpose": transpose,
        "normalize_incoming": normalize_incoming,
        "ones_first_hop_range": [float(first_hop.min()), float(first_hop.max())],
        "ones_second_hop_range": [float(second_hop.min()), float(second_hop.max())],
        "node_order": "sensors.csv file order, without sorting",
        "node_alignment": "The release arrays have no independent station labels",
        "model_orientation": "A[source, destination]",
    }


def _check_split_overlap(windows, frames):
    """Record equality in the release without inferring absolute timestamps."""
    report = {}
    for left, right in combinations(windows, 2):
        same_windows = windows[left].keys() & windows[right].keys()
        same_frames = frames[left].keys() & frames[right].keys()
        report[f"{left}/{right}"] = {
            "matching_history_and_target_windows": len(same_windows),
            "matching_target_frames": len(same_frames),
            "example_window_ids": sorted(
                (windows[left][key], windows[right][key]) for key in same_windows
            )[:5],
            "example_target_frame_ids_and_steps": sorted(
                (frames[left][key], frames[right][key]) for key in same_frames
            )[:5],
        }
    return report


def prepare_xtraffic(data_dir, output_dir, config):
    """Validate the trusted release and prepare shared splits, scale and support."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    if config["model"] != MODEL_VARIANT:
        raise ValueError(f"This branch expects XTraffic model: {MODEL_VARIANT}")
    args = config["model_args"]
    num_nodes = args["num_nodes"]
    in_steps = args["in_steps"]
    out_steps = args["out_steps"]
    steps_per_day = args["steps_per_day"]
    missing_value = config.get("missing_traffic_value")
    if missing_value is not None and (
        type(missing_value) not in (int, float) or not np.isfinite(missing_value)
    ):
        raise ValueError("missing_traffic_value must be null or a finite original-scale value")

    release_scale = _read_release_scale(data_dir / "incident_stats.npz")
    targets_normalized = config.get("targets_normalized")
    if targets_normalized is None:
        targets_normalized = release_scale[0]
    if not isinstance(targets_normalized, bool):
        raise ValueError("targets_normalized must be null, true or false")
    type_names = _read_type_names(data_dir / "type_mapping.json")
    adjacency, graph_report = read_road_graph(data_dir, num_nodes, **config.get("road_graph", {}))

    sample_path = data_dir / "incident_all.npy"
    # The IGSTGNN release serializes its dictionaries in a NumPy object array.
    samples = np.load(sample_path, allow_pickle=True)
    if not isinstance(samples, np.ndarray) or samples.ndim != 1 or samples.dtype.kind != "O":
        raise ValueError("incident_all.npy must be a one-dimensional object array from IGSTGNN")
    splits = released_splits(len(samples))
    masks = np.empty((len(samples), num_nodes), dtype=bool)
    types = np.empty(len(samples), dtype=np.int64)
    moments = _TrafficMoments()
    position_counts = Counter()
    coverage_counts = Counter()
    sample_errors = []
    traffic_report = {}
    window_hashes = {name: {} for name in splits}
    frame_hashes = {name: {} for name in splits}

    for split, sample_ids in splits.items():
        invalid_history = invalid_target = 0
        history_min = target_min = float("inf")
        history_max = target_max = float("-inf")
        for sample_id in sample_ids:
            try:
                history, future, mask, event_type, position = _sample_fields(
                    samples[sample_id], in_steps=in_steps, out_steps=out_steps,
                    num_nodes=num_nodes, steps_per_day=steps_per_day, type_names=type_names,
                )
            except (KeyError, ValueError) as error:
                sample_errors.append({"sample_id": sample_id, "split": split, "reason": str(error)})
                continue
            masks[sample_id] = mask
            types[sample_id] = event_type
            coverage_counts[int(mask.sum())] += 1
            position_counts[position] += 1
            history_raw, history_valid = _restore_traffic(
                history[..., 0], *release_scale, missing_value,
            )
            target_raw, target_valid = _restore_traffic(
                future[..., 0], targets_normalized, *release_scale[1:], missing_value,
            )
            valid_history = history_raw[history_valid]
            valid_target = target_raw[target_valid]
            invalid_history += history_valid.size - valid_history.size
            invalid_target += target_valid.size - valid_target.size
            if valid_history.size:
                history_min = min(history_min, float(valid_history.min()))
                history_max = max(history_max, float(valid_history.max()))
            if valid_target.size:
                target_min = min(target_min, float(valid_target.min()))
                target_max = max(target_max, float(valid_target.max()))
            if split == "train":
                moments.update(valid_history)

            window = (_fingerprint(history), _fingerprint(future))
            window_hashes[split].setdefault(window, sample_id)
            for step, frame in enumerate(future):
                frame_hashes[split].setdefault(_fingerprint(frame), (sample_id, step))
        traffic_report[split] = {
            "history_raw_range": [history_min, history_max] if np.isfinite(history_min) else None,
            "target_raw_range": [target_min, target_max] if np.isfinite(target_min) else None,
            "invalid_history_values": int(invalid_history),
            "invalid_target_values": int(invalid_target),
        }

    report = {
        "release_scale": {
            "history_normalized": release_scale[0], "targets_normalized": targets_normalized,
            "mean": release_scale[1], "std": release_scale[2], "missing_traffic_value": missing_value,
        },
        "traffic": traffic_report,
        "graph": graph_report,
        "event_category_ids": EVENT_TYPE_IDS,
        "event_positions": dict(sorted(position_counts.items())),
        "selected_nodes_per_sample": dict(sorted(coverage_counts.items())),
        "all_zero_masks": coverage_counts[0],
        "all_one_masks": coverage_counts[num_nodes],
        "candidate_cross_split_overlap": _check_split_overlap(window_hashes, frame_hashes),
        "timing": "Release positions do not establish report timestamps or absolute split boundaries",
        "sample_errors": sample_errors,
    }
    audit_path = output_dir / "audit.json"
    if sample_errors:
        save_json(audit_path, report)
        raise ValueError(f"{len(sample_errors)} malformed samples; see {audit_path}")

    scale = moments.statistics()
    report["training_scale"] = scale
    manifest = {
        "region": config["region"],
        "file": sample_path.name,
        "sha256": _sha256(sample_path),
        "splits": {name: list(indices) for name, indices in splits.items()},
    }
    save_json(audit_path, report)
    save_json(output_dir / "split_manifest.json", manifest)
    dataset = XTrafficDataset(
        samples, masks, types, steps_per_day, release_scale,
        targets_normalized, scale, missing_value,
    )
    return dataset, splits, adjacency, report
