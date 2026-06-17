"""
Checkpoint system for LLM experiments.

Saves predictions line by line in JSONL. If the process crashes, the runner resumes
from the last saved index without repeating calls already made.

File structure:
  checkpoints/{model_alias}_{strategy}_{split}.jsonl
    -> one line per instance: {"idx": int, "label": int, "pred": int|null, ...}
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("ftr_llm.checkpoint")

# Global lock per checkpoint file — ensures that concurrent writes
# from multiple coroutines do not interleave on the same JSONL line.
_file_locks: dict[str, asyncio.Lock] = {}


def _get_lock(path: Path) -> asyncio.Lock:
    key = str(path)
    if key not in _file_locks:
        _file_locks[key] = asyncio.Lock()
    return _file_locks[key]


def checkpoint_path(ckpt_dir: Path, model_alias: str, strategy: str, split: str) -> Path:
    return ckpt_dir / f"{model_alias}__{strategy}__{split}.jsonl"


def load_checkpoint(path: Path) -> dict[int, dict]:
    """Returns {idx: record} with all already-saved entries."""
    done: dict[int, dict] = {}
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done[rec["idx"]] = rec
            except Exception:
                pass
    logger.info("Checkpoint loaded: %d records in %s", len(done), path.name)
    return done


def append_record(path: Path, record: dict) -> None:
    """Appends a record to the checkpoint JSONL (atomic write per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


async def async_append_record(path: Path, record: dict) -> None:
    """Async-safe version of append_record — uses per-file lock."""
    async with _get_lock(path):
        await asyncio.to_thread(append_record, path, record)


def checkpoint_to_lists(done: dict[int, dict], n_total: int) -> tuple[list, list, list[int]]:
    """Converts checkpoint into (y_true, y_pred, failed_idxs)."""
    y_true, y_pred = [], []
    failed = []
    for i in range(n_total):
        rec = done.get(i)
        if rec is None:
            continue
        if rec.get("pred") is None:
            failed.append(i)
            continue
        y_true.append(rec["label"])
        y_pred.append(rec["pred"])
    return y_true, y_pred, failed
