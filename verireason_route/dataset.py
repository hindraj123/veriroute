"""Dataset loading, normalization, and auditable quality filtering."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .data import RoutingTask
from .verifier import RoutingVerifier


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("invalid JSON on line %d: %s" % (number, exc))
            if not isinstance(row, dict):
                raise ValueError("line %d must contain a JSON object" % number)
            rows.append(row)
    return rows


def normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Accept common instruction/response and prompt/completion JSONL shapes."""
    prompt = row.get("prompt") or row.get("instruction") or row.get("input")
    target = row.get("completion") or row.get("response") or row.get("output") or row.get("target")
    if isinstance(row.get("messages"), list):
        messages = row["messages"]
        prompt = "\n".join(str(m.get("content", "")) for m in messages[:-1]
                            if isinstance(m, dict) and m.get("role") in ("system", "user"))
        target = messages[-1].get("content") if messages and isinstance(messages[-1], dict) else target
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("missing prompt/instruction/input")
    if not isinstance(target, str) or not target.strip():
        raise ValueError("missing completion/response/output")
    normalized = {"prompt": prompt.strip(), "completion": target.strip()}
    if isinstance(row.get("task"), dict):
        normalized["task"] = row["task"]
    if isinstance(row.get("metrics"), dict):
        normalized["metrics"] = row["metrics"]
    return normalized


def filter_rows(rows: Iterable[Dict[str, Any]], min_prompt_chars: int = 20,
                min_completion_chars: int = 8, max_completion_chars: int = 20000,
                verifier: Optional[RoutingVerifier] = None,
                require_legal_route: bool = True) -> Tuple[List[Dict[str, Any]], Counter]:
    """Drop malformed/duplicate/invalid examples and return reason counts."""
    verifier = verifier or RoutingVerifier()
    kept: List[Dict[str, Any]] = []
    counts: Counter = Counter()
    seen = set()
    for index, raw in enumerate(rows):
        try:
            row = normalize_row(raw)
        except ValueError as exc:
            counts["schema:" + str(exc)] += 1
            continue
        if len(row["prompt"]) < min_prompt_chars:
            counts["prompt_too_short"] += 1
            continue
        if not min_completion_chars <= len(row["completion"]) <= max_completion_chars:
            counts["completion_length"] += 1
            continue
        key = (row["prompt"], row["completion"])
        if key in seen:
            counts["duplicate"] += 1
            continue
        if require_legal_route:
            task_data = row.get("task")
            if not isinstance(task_data, dict):
                counts["missing_verifier_task"] += 1
                continue
            try:
                task = RoutingTask.from_dict(task_data)
                evaluation = verifier.evaluate(task, row["completion"])
            except (KeyError, TypeError, ValueError):
                counts["invalid_task"] += 1
                continue
            if not evaluation.legal:
                counts["route_not_legal"] += 1
                continue
        seen.add(key)
        row["example_id"] = str(raw.get("example_id", raw.get("id", index)))
        kept.append(row)
        counts["kept"] += 1
    return kept, counts
