"""
Module 3 - Rare disease phenotype matching. Not built yet (weeks 9-11).

Stub for orchestrator routing; see src/modules/repurposing/__init__.py for why
these exist now rather than later.

Ethical framing, stated here because it belongs in the code and not only in
the README: when built, this is a *research triage aid*. It ranks candidate
diagnoses for a researcher to investigate. It is not a diagnostic tool and
must never be presented as one.
"""
from __future__ import annotations

from src.schemas import ModuleResult, ModuleStatus


def match_phenotypes(symptom_description: str, top_k: int = 10) -> ModuleResult:
    """Planned: free text -> HPO terms -> ranked differential over phenotype.hpoa."""
    return ModuleResult(
        module="phenotype",
        status=ModuleStatus.NOT_IMPLEMENTED,
        notes=(
            "Module 3 (rare disease phenotype matching) is not implemented yet - "
            "scheduled for weeks 9-11. No differential can be returned. "
            "Do not substitute background knowledge for a result."
        ),
    )
