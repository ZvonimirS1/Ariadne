"""
Structured run logging.

Week 14 asks for hallucination rates, token costs, and ablation comparisons.
Reconstructing those from console scrollback at the end of the semester is
miserable, so every run appends machine-readable events to a JSONL file as it
happens. The eval harness reads these back; so will the final writeup.

JSONL specifically because it survives a crashed run - each line is flushed
independently, so a partial log is still a readable log.
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src import config


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def progress(message: str) -> None:
    """
    Human-readable progress on stderr.

    A single question can mean 20+ LLM calls at ~10s each, and a terminal that
    shows nothing for two minutes is indistinguishable from a hang. stderr keeps
    `cli.py --json` output clean. Set ARIADNE_QUIET=1 to silence it.
    """
    if os.getenv("ARIADNE_QUIET"):
        return
    print(message, file=sys.stderr, flush=True)


class RunLog:
    """
    One run = one JSONL file. Usable as a context manager:

        with RunLog("cli") as log:
            log.event("question", text=question)
    """

    def __init__(self, label: str = "run", run_id: str | None = None):
        config.ensure_dirs()
        self.run_id = run_id or f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
        self.label = label
        self.path: Path = config.RUN_LOG_DIR / f"{self.run_id}.jsonl"
        self._started = time.perf_counter()
        self._totals: dict[str, int] = {}
        self.event("run_start", label=label, model=config.ANTHROPIC_MODEL)

    def event(self, kind: str, **fields: Any) -> None:
        record = {
            "ts": _utc_now(),
            "run_id": self.run_id,
            "kind": kind,
            "elapsed": round(time.perf_counter() - self._started, 3),
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def add_usage(self, usage: dict[str, int]) -> None:
        """Accumulate token usage across every call in the run."""
        for key, value in usage.items():
            self._totals[key] = self._totals.get(key, 0) + int(value or 0)

    @property
    def totals(self) -> dict[str, int]:
        return dict(self._totals)

    def close(self, **fields: Any) -> None:
        self.event("run_end", usage_totals=self.totals, **fields)

    def __enter__(self) -> "RunLog":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is not None:
            self.event("error", error_type=exc_type.__name__, message=str(exc))
        self.close(ok=exc is None)
