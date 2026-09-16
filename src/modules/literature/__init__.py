"""
Module 1 - Literature Synthesis. The full pipeline in one place.

    search PubMed -> fetch abstracts -> extract findings -> VERIFY -> synthesize

The verification step in the middle is not optional and has no bypass. That is
the whole point: there is no code path from "the model said something" to "the
system claims something" that skips provenance checking.
"""
from __future__ import annotations

import time

from src.modules.literature.extract import extract_all
from src.modules.literature.pubmed_client import PubMedError, search_and_fetch
from src.modules.literature.synthesize import synthesize
from src.provenance import verify_all
from src.schemas import EvidenceGroup, ModuleResult, ModuleStatus
from src.telemetry import progress


def run_literature_module(
    question: str,
    *,
    max_results: int = 10,
    min_year: int | None = None,
    search_query: str | None = None,
    write_summary: bool = True,
) -> tuple[ModuleResult, list[EvidenceGroup]]:
    """
    Full pipeline. Returns the result plus the evidence groups (which the eval
    harness scores contradiction detection against).

    `search_query` lets the caller pass a PubMed-shaped query distinct from the
    natural-language question - a question mark and ordinary English words make
    for a poor E-utilities query.
    """
    started = time.perf_counter()
    progress(f"  searching PubMed: {search_query or question!r}")

    try:
        abstracts = search_and_fetch(
            search_query or question,
            max_results=max_results,
            min_year=min_year,
        )
    except PubMedError as exc:
        return (
            ModuleResult(
                module="literature",
                status=ModuleStatus.ERROR,
                notes=f"PubMed request failed: {exc}",
                elapsed_seconds=round(time.perf_counter() - started, 3),
            ),
            [],
        )

    if not abstracts:
        return (
            ModuleResult(
                module="literature",
                status=ModuleStatus.NO_RESULTS,
                notes=f"no PubMed records matched {search_query or question!r}",
                elapsed_seconds=round(time.perf_counter() - started, 3),
            ),
            [],
        )

    progress(f"  retrieved {len(abstracts)} abstracts, extracting findings (~10s per call)...")
    candidates, extract_usage = extract_all(abstracts)

    # The chokepoint. Everything downstream sees only what survived this.
    report = verify_all(candidates, abstracts)
    progress(f"  grounding: {report.summary_line()}")

    result, groups = synthesize(question, report, write_summary=write_summary)

    # Roll extraction cost into the module's usage - it dominates the total,
    # so reporting only the summary call would badly understate the run.
    merged = dict(result.usage)
    for key, value in extract_usage.items():
        merged[key] = merged.get(key, 0) + value
    result.usage = merged
    result.elapsed_seconds = round(time.perf_counter() - started, 3)
    result.notes = "; ".join(
        part for part in (f"{len(abstracts)} abstracts retrieved", result.notes) if part
    )

    return result, groups
