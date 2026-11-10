"""Composable reward terms for verifier-guided routing GRPO.

Each component returns a bounded score in [-1, 1]. Physical DRC/congestion
terms can be supplied in task metadata by a real EDA backend later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from .data import RoutingTask
from .verifier import RoutingVerifier, extract_json


@dataclass(frozen=True)
class RewardWeights:
    format: float = 0.10
    connectivity: float = 0.20
    legality: float = 0.40
    wirelength: float = 0.20
    drc: float = 0.10


class RoutingReward:
    """Weighted, inspectable reward with independent diagnostic components."""

    def __init__(self, weights: RewardWeights = RewardWeights(), verifier: Optional[RoutingVerifier] = None):
        self.weights = weights
        self.verifier = verifier or RoutingVerifier()

    def components(self, task: RoutingTask, completion: str,
                   external_metrics: Optional[Mapping[str, float]] = None) -> Dict[str, float]:
        result = self.verifier.evaluate(task, completion)
        parsed = extract_json(completion)
        fmt = 1.0 if parsed is not None and isinstance(parsed.get("path"), list) else -1.0
        connected = 1.0 if result.connected else -1.0
        legal = 1.0 if result.legal else -1.0
        if result.legal and result.path_length is not None and result.optimal_length is not None:
            wirelength = max(-1.0, 1.0 - (result.path_length - result.optimal_length) /
                             float(max(1, result.optimal_length)))
        else:
            wirelength = -1.0
        external = external_metrics or {}
        # A backend reports nonnegative violations. Absence of metrics is neutral;
        # this prevents synthetic data from pretending to be physical DRC evidence.
        drc_count = external.get("drc_violations")
        drc = (1.0 if float(drc_count) == 0 else -1.0) if drc_count is not None else 0.0
        return {"format": fmt, "connectivity": connected, "legality": legal,
                "wirelength": wirelength, "drc": drc}

    def score(self, task: RoutingTask, completion: str,
              external_metrics: Optional[Mapping[str, float]] = None) -> float:
        parts = self.components(task, completion, external_metrics)
        weights = self.weights
        pairs = ((weights.format, parts["format"]), (weights.connectivity, parts["connectivity"]),
                 (weights.legality, parts["legality"]), (weights.wirelength, parts["wirelength"]),
                 (weights.drc, parts["drc"]))
        denominator = sum(abs(weight) for weight, _ in pairs)
        return sum(weight * value for weight, value in pairs) / denominator if denominator else 0.0


class CircuitNetCongestionReward:
    """GRPO reward for a predicted congestion patch against hidden label data."""

    def __init__(self, format_weight: float = 0.1, map_weight: float = 0.65,
                 hotspot_weight: float = 0.25):
        self.weights = {"format": format_weight, "map": map_weight, "hotspot": hotspot_weight}

    def components(self, task: Mapping[str, Any], completion: str) -> Dict[str, float]:
        parsed = extract_json(completion)
        target = task.get("target")
        bins = int(task.get("quantization_bins", 16))
        if not isinstance(parsed, dict) or not isinstance(parsed.get("congestion"), list):
            return {"format": -1.0, "map": -1.0, "hotspot": -1.0}
        pred = parsed["congestion"]
        if not isinstance(target, list) or len(pred) != len(target) or not target:
            return {"format": -1.0, "map": -1.0, "hotspot": -1.0}
        errors, pred_hot, true_hot, tp = [], 0, 0, 0
        threshold = max(1, int(math.ceil((bins - 1) * 0.75)))
        for pred_row, true_row in zip(pred, target):
            if not isinstance(pred_row, list) or len(pred_row) != len(true_row):
                return {"format": -1.0, "map": -1.0, "hotspot": -1.0}
            for p, t in zip(pred_row, true_row):
                if not isinstance(p, (int, float)) or not isinstance(t, (int, float)):
                    return {"format": -1.0, "map": -1.0, "hotspot": -1.0}
                p, t = int(p), int(t)
                if not 0 <= p < bins or not 0 <= t < bins:
                    return {"format": -1.0, "map": -1.0, "hotspot": -1.0}
                errors.append(abs(p - t) / float(max(1, bins - 1)))
                ph, th = p >= threshold, t >= threshold
                pred_hot += int(ph)
                true_hot += int(th)
                tp += int(ph and th)
        mae = sum(errors) / max(1, len(errors))
        map_score = 1.0 - 2.0 * mae
        if pred_hot == 0 and true_hot == 0:
            hotspot_f1 = 1.0
        else:
            precision = tp / float(max(1, pred_hot))
            recall = tp / float(max(1, true_hot))
            hotspot_f1 = 2 * precision * recall / max(1e-12, precision + recall)
        return {"format": 1.0, "map": map_score, "hotspot": 2.0 * hotspot_f1 - 1.0}

    def score(self, task: Mapping[str, Any], completion: str) -> float:
        parts = self.components(task, completion)
        denom = sum(abs(w) for w in self.weights.values())
        return sum(self.weights[k] * parts[k] for k in self.weights) / denom if denom else 0.0


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        return "".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in completion)
    return str(completion)
