"""
Module 2 - Drug repurposing over a knowledge graph. Not built yet (week 6-8).

This stub exists so the orchestrator can route to it today. A routing bug is
much easier to find now, against a module that returns a clear
"not_implemented", than later against one that returns plausible prose.

The important property: when asked a repurposing question, the system says it
cannot answer rather than inventing one. That is the behaviour under test.
"""
from __future__ import annotations

from src.schemas import ModuleResult, ModuleStatus


def find_repurposing_candidates(disease_or_drug: str, max_candidates: int = 10) -> ModuleResult:
    """Planned: metapath search over Hetionet (drug -> gene -> disease), then ranking."""
    return ModuleResult(
        module="repurposing",
        status=ModuleStatus.NOT_IMPLEMENTED,
        notes=(
            "Module 2 (drug repurposing over Hetionet) is not implemented yet - "
            "scheduled for weeks 6-8. No candidates can be returned for "
            f"{disease_or_drug!r}. Do not substitute background knowledge for a result."
        ),
    )
