"""Independent route parser, legality checker, and shaped reward."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .data import Point, RoutingTask, shortest_path


@dataclass
class Evaluation:
    reward: float
    valid_json: bool
    connected: bool
    legal: bool
    path_length: Optional[int]
    optimal_length: Optional[int]
    violations: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    # Accept a single fenced JSON answer while rejecting unrelated prose.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    candidate = fenced.group(1) if fenced else text
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except (ValueError, TypeError):
        return None


class RoutingVerifier:
    def evaluate(self, task: RoutingTask, completion: str) -> Evaluation:
        violations: List[str] = []
        obj = extract_json(completion)
        if obj is None:
            return Evaluation(-1.0, False, False, False, None, None, ["invalid_json"])
        raw = obj.get("path")
        if not isinstance(raw, list) or not raw:
            return Evaluation(-0.7, True, False, False, None, None, ["missing_path"])
        try:
            path = [(int(p[0]), int(p[1])) for p in raw if isinstance(p, (list, tuple)) and len(p) == 2]
        except (ValueError, TypeError, IndexError):
            path = []
        if len(path) != len(raw):
            violations.append("malformed_coordinate")
        if len(set(path)) != len(path):
            violations.append("self_intersection")
        connected = bool(path) and path[0] == task.source and path[-1] == task.target
        if not connected:
            violations.append("wrong_endpoints")
        occupied = {p for route in task.existing_routes for p in route}
        for i, (r, c) in enumerate(path):
            if not (0 <= r < task.rows and 0 <= c < task.cols):
                violations.append("out_of_bounds")
                continue
            if task.grid[r][c] == "#":
                violations.append("obstacle_collision")
            if (r, c) in occupied:
                violations.append("net_conflict")
            if i and abs(r - path[i - 1][0]) + abs(c - path[i - 1][1]) != 1:
                violations.append("non_orthogonal_step")
        violations = sorted(set(violations))
        legal = connected and not violations
        ref = shortest_path(task)
        optimal = len(ref) - 1 if ref else None
        length = len(path) - 1 if path else None
        if legal and length is not None and optimal is not None:
            excess = max(0, length - optimal)
            reward = 1.0 - min(0.5, 0.025 * excess)
        else:
            penalty = 0.18 * len(violations)
            reward = -0.25 - min(0.65, penalty)
            if connected:
                reward += 0.1
        return Evaluation(
            reward=reward,
            valid_json=True,
            connected=connected,
            legal=legal,
            path_length=length,
            optimal_length=optimal,
            violations=violations,
            metrics={"wirelength": float(length if length is not None else math.inf),
                     "optimal_wirelength": float(optimal if optimal is not None else math.inf)},
        )

    def repair_candidate(self, task: RoutingTask, reasoning: str = "Use the shortest legal route.") -> Optional[str]:
        path = shortest_path(task)
        if path is None:
            return None
        return json.dumps({"reasoning": reasoning, "path": [list(p) for p in path]})
