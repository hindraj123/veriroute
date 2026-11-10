"""Prepare CircuitNet congestion maps as grouped tile-patch SFT examples.

Input is the `feature/` and `label/` output from CircuitNet's official
`generate_training_set.py --task congestion` preprocessing: HxWx3 normalized
feature maps and HxWx1 congestion maps, matched by filename.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Tuple


CHANNELS = ("macro_region", "RUDY", "pin_RUDY")


def _load_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit("CircuitNet preparation needs numpy; install with `pip install numpy`") from exc
    return np


def _normalize_shapes(features, labels, filename: str):
    if features.ndim != 3:
        raise ValueError("%s feature array must be HxWxC, got %r" % (filename, features.shape))
    if labels.ndim == 2:
        labels = labels[:, :, None]
    if labels.ndim != 3 or labels.shape[2] != 1:
        raise ValueError("%s congestion label must be HxWx1, got %r" % (filename, labels.shape))
    if features.shape[:2] != labels.shape[:2]:
        raise ValueError("%s feature and label map sizes differ" % filename)
    if features.shape[2] != len(CHANNELS):
        raise ValueError("%s expected %d feature channels (%s), got %d" %
                         (filename, len(CHANNELS), ", ".join(CHANNELS), features.shape[2]))
    return features, labels


def _quantize(array, bins: int):
    # CircuitNet's official packer min-max normalizes each map. Clip defensively
    # and use integer levels to keep the serialized prompt compact.
    return int(round(max(0.0, min(1.0, float(array))) * (bins - 1)))


def _matrix_rows(patch, bins: int) -> str:
    return "\n".join(" ".join(str(_quantize(v, bins)) for v in row) for row in patch)


def _example(sample: str, y: int, x: int, features, labels, bins: int, patch_size: int) -> dict:
    ph = pw = patch_size
    channels = []
    for index, channel in enumerate(CHANNELS):
        channels.append("%s:\n%s" %
                        (channel, _matrix_rows(features[y:y + ph, x:x + pw, index], bins)))
    prompt = (
        "Task: predict the global-routing congestion map for this CircuitNet layout patch.\n"
        "Feature values and target values are quantized integers from 0 to %d.\n"
        "Patch size: %dx%d. Channel order is macro_region, RUDY, pin_RUDY.\n"
        "Return only JSON: {\"congestion\":[[...], ...]} with exactly %d rows and %d values per row.\n"
        "Layout sample: %s; patch origin: row %d, column %d.\n%s"
    ) % (bins - 1, ph, pw, ph, pw, sample, y, x, "\n".join(channels))
    target = labels[y:y + ph, x:x + pw, 0]
    completion = json.dumps({"congestion": [[_quantize(v, bins) for v in row] for row in target]},
                            separators=(",", ":"))
    task_id = "%s:r%d:c%d" % (sample, y, x)
    return {"prompt": prompt, "completion": completion, "example_id": task_id,
            # GRPO sees the reference only inside its reward function; it is
            # never included in the model prompt.
            "task": {"task_type": "circuitnet_congestion", "task_id": task_id,
                     "target": [[_quantize(v, bins) for v in row] for row in target],
                     "quantization_bins": bins}}


def prepare(feature_dir: str, label_dir: str, output_dir: str, patch_size: int = 8,
            stride: int = 8, bins: int = 16, eval_fraction: float = 0.1,
            seed: int = 17, max_samples: int = 20000) -> Tuple[int, int]:
    np = _load_numpy()
    if patch_size < 1 or stride < 1 or bins < 2:
        raise ValueError("patch_size/stride must be positive and bins must be >= 2")
    if not 0 <= eval_fraction < 1:
        raise ValueError("eval_fraction must be in [0, 1)")
    feature_root, label_root = Path(feature_dir), Path(label_dir)
    files = sorted(feature_root.glob("*.npy"))
    if not files:
        raise ValueError("No .npy feature files found in %s" % feature_root)
    pairs = []
    for feature_path in files:
        label_path = label_root / feature_path.name
        if label_path.is_file():
            pairs.append((feature_path, label_path))
    if not pairs:
        raise ValueError("No matching feature/label .npy filenames found")

    # Split by whole layout sample before crop generation. Crops of one layout
    # can therefore never land in both train and eval.
    rng = random.Random(seed)
    order = list(range(len(pairs)))
    rng.shuffle(order)
    eval_count = 0 if len(order) < 2 or eval_fraction == 0 else max(1, int(round(len(order) * eval_fraction)))
    eval_count = min(eval_count, len(order) - 1)
    pairs = [pairs[index] for index in order]
    eval_ids = set(range(eval_count))
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    writers = {
        "train": (out_root / "train.jsonl").open("w", encoding="utf-8"),
        "eval": (out_root / "eval.jsonl").open("w", encoding="utf-8"),
    }
    counts = {"train": 0, "eval": 0}
    try:
        for pair_index, (feature_path, label_path) in enumerate(pairs):
            split = "eval" if pair_index in eval_ids else "train"
            if max_samples and counts[split] >= max_samples:
                continue
            features = np.load(str(feature_path), mmap_mode="r")
            labels = np.load(str(label_path), mmap_mode="r")
            features, labels = _normalize_shapes(features, labels, feature_path.name)
            height, width, _ = features.shape
            sample = feature_path.stem
            for y in range(0, height - patch_size + 1, stride):
                for x in range(0, width - patch_size + 1, stride):
                    row = _example(sample, y, x, features, labels, bins, patch_size)
                    writers[split].write(json.dumps(row, separators=(",", ":")) + "\n")
                    counts[split] += 1
                    if max_samples and counts[split] >= max_samples:
                        break
                if max_samples and counts[split] >= max_samples:
                    break
    finally:
        for handle in writers.values():
            handle.close()
    if not counts["train"]:
        raise ValueError("No train patches generated; check patch_size and dataset dimensions")
    return counts["train"], counts["eval"]


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--features", required=True, help="CircuitNet prepared feature/ directory")
    parser.add_argument("--labels", required=True, help="CircuitNet prepared label/ directory")
    parser.add_argument("--output", default="data/circuitnet-congestion")
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-samples", type=int, default=20000,
                        help="Maximum patches per split for a manageable pilot; 0 means all patches")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    args = parser.parse_args()
    train_count, eval_count = prepare(args.features, args.labels, args.output,
                                      args.patch_size, args.stride, args.bins,
                                      args.eval_fraction, args.seed, args.max_samples)
    print("wrote %d train and %d eval patches under %s" % (train_count, eval_count, args.output))
