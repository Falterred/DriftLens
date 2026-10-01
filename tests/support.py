"""Shared helper functions for building test inputs."""

from __future__ import annotations

import json
import random
from pathlib import Path


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    """Write `rows` as JSON Lines with UTF-8 text and `\n` terminators."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    return target


def make_rows(n: int, seed: int, dim: int | None = None) -> list[dict]:
    """Build `n` deterministic rows, optionally carrying a float embedding."""
    generator = random.Random(seed)
    rows: list[dict] = []
    for i in range(n):
        chunk_index = 0
        row: dict = {
            "record_id": f"d{i}-c{chunk_index}",
            "text": f"word{generator.randrange(1_000_000)}",
        }
        if dim is not None:
            row["embedding"] = [generator.random() for _ in range(dim)]
        rows.append(row)
    return rows
