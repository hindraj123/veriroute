"""Task representation and small deterministic synthetic task generator."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

Point = Tuple[int, int]


@dataclass
class RoutingTask:
    grid: List[str]
    source: Point
    target: Point
    existing_routes: List[List[Point]]
    task_id: str = "task"

    @property
    def rows(self) -> int:
        return len(self.grid)

    @property
    def cols(self) -> int:
        return len(self.grid[0]) if self.grid else 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "grid": self.grid,
            "source": list(self.source),
            "target": list(self.target),
            "existing_routes": [[list(p) for p in route] for route in self.existing_routes],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoutingTask":
        return cls(
            grid=list(data["grid"]),
            source=tuple(data["source"]),
            target=tuple(data["target"]),
            existing_routes=[[tuple(p) for p in route] for route in data.get("existing_routes", [])],
            task_id=str(data.get("task_id", "task")),
        )

    def prompt(self) -> str:
        occupied = {p for route in self.existing_routes for p in route}
        board = []
        for r, row in enumerate(self.grid):
            chars = []
            for c, char in enumerate(row):
                p = (r, c)
                chars.append("S" if p == self.source else "T" if p == self.target else "R" if p in occupied else char)
            board.append("".join(chars))
        return (
            "Route one signal on this orthogonal grid. Return exactly one JSON object with keys "
            '"reasoning" (short design rationale) and "path" (ordered [row,column] coordinates). '
            "The path must start at source, end at target, stay in bounds, use only free cells, "
            "move one cell orthogonally per step, avoid # obstacles and R existing routes.\n"
            "Grid legend: # obstacle, . free, S source, T target, R occupied.\n"
            f"Source={list(self.source)} Target={list(self.target)}\nGrid:\n" + "\n".join(board)
        )


def shortest_path(task: RoutingTask) -> Optional[List[Point]]:
    """BFS reference route; respects obstacles and previously routed nets."""
    from collections import deque

    blocked = {p for route in task.existing_routes for p in route}
    blocked.discard(task.source)
    blocked.discard(task.target)
    queue = deque([task.source])
    parent: Dict[Point, Optional[Point]] = {task.source: None}
    while queue:
        p = queue.popleft()
        if p == task.target:
            path = []
            while p is not None:
                path.append(p)
                p = parent[p]  # type: ignore[assignment]
            return list(reversed(path))
        r, c = p
        for nxt in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            nr, nc = nxt
            if (0 <= nr < task.rows and 0 <= nc < task.cols and task.grid[nr][nc] != "#"
                    and nxt not in blocked and nxt not in parent):
                parent[nxt] = p
                queue.append(nxt)
    return None


def generate_task(rng: random.Random, size: int = 8, obstacle_rate: float = 0.18, index: int = 0) -> RoutingTask:
    if size < 4:
        raise ValueError("size must be at least 4")
    for _ in range(100):
        src = (rng.randrange(size), rng.randrange(size))
        dst = (rng.randrange(size), rng.randrange(size))
        while dst == src:
            dst = (rng.randrange(size), rng.randrange(size))
        grid = [["#" if rng.random() < obstacle_rate else "." for _ in range(size)] for _ in range(size)]
        grid[src[0]][src[1]] = "."
        grid[dst[0]][dst[1]] = "."
        task = RoutingTask(["".join(row) for row in grid], src, dst, [], "synthetic-%05d" % index)
        if shortest_path(task):
            return task
    raise RuntimeError("could not generate a routable task")


def task_json(task: RoutingTask) -> str:
    return json.dumps(task.to_dict(), separators=(",", ":"))
