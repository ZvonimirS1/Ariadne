"""
Runs an eval dataset through a module and scores it.

Calls the module pipeline directly rather than going through the orchestrator,
so a bad score points at extraction or grounding rather than at routing. Route
quality is tested separately in tests/test_routing.py.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from eval.metrics import EvalReport, QuestionScore, score_question
from src.config import PROJECT_ROOT
from src.modules.literature import run_literature_module
from src.schemas import ModuleStatus
from src.telemetry import RunLog

DATASET_DIR = PROJECT_ROOT / "eval" / "datasets"


def load_dataset(name: str) -> list[dict]:
    path = DATASET_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no eval dataset at {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data["questions"]


def run_literature_eval(
    *,
    limit: int | None = None,
    max_results: int = 10,
    write_summary: bool = False,
    log: RunLog | None = None,
) -> EvalReport:
    """
    Score the literature module against its dataset.

    `write_summary` defaults to False: the prose summary is presentation, and
    none of the metrics read it, so skipping it removes an LLM call per
    question at no cost to the measurement.
    """
    specs = load_dataset("literature")
    if limit:
        specs = specs[:limit]

    owns_log = log is None
    log = log or RunLog("eval-literature")
    report = EvalReport(module="literature")

    for index, spec in enumerate(specs, start=1):
        print(f"  [{index}/{len(specs)}] {spec['id']} ...", flush=True)
        try:
            result, groups = run_literature_module(
                question=spec["question"],
                search_query=spec.get("search_query"),
                max_results=max_results,
                write_summary=write_summary,
            )
            score = score_question(spec, result, groups)
        except Exception as exc:  # one bad question shouldn't void the whole run
            score = QuestionScore(
                id=spec["id"],
                status=ModuleStatus.ERROR.value,
                expected_direction=spec["expected_direction"],
                notes=f"{type(exc).__name__}: {exc}",
            )

        report.scores.append(score)
        log.add_usage(score.usage)
        log.event("eval_question", **score.model_dump(mode="json"))

    log.event(
        "eval_summary",
        direction_accuracy=report.direction_accuracy,
        hallucination_rate=report.hallucination_rate,
        contradiction_prf=report.contradiction_prf(),
        estimated_cost_usd=round(report.estimated_cost_usd, 4),
    )
    if owns_log:
        log.close(ok=True)
    return report
