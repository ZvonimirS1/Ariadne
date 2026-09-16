"""
The tool surface the orchestrator sees.

Each tool wraps a module and renders its ModuleResult as text for the model to
read. The structured ModuleResult is *also* kept in a collector, because the
model's prose is a rendering and the authoritative grounded claims are the
typed objects. The CLI and the eval harness read the collector, not the prose.

Docstrings here are load-bearing: the SDK generates each tool's description
and JSON schema from the signature and docstring, so they are prompt text, not
just documentation.

Threading note: the collector is module-level, which is fine for the CLI and
eval harness (one run at a time). Serving concurrent runs in one process would
need this moved to a contextvar.
"""
from __future__ import annotations

from anthropic import beta_tool

from src.modules.phenotype import match_phenotypes as _match_phenotypes
from src.modules.repurposing import find_repurposing_candidates as _find_repurposing
from src.schemas import ModuleResult, ModuleStatus

_collected: list[ModuleResult] = []


def reset_collector() -> None:
    _collected.clear()


def collected() -> list[ModuleResult]:
    """Structured results from every tool call in the current run."""
    return list(_collected)


def _render(result: ModuleResult) -> str:
    """Render a ModuleResult as the text the model reads back as a tool result."""
    if result.status is ModuleStatus.NOT_IMPLEMENTED:
        return f"NOT IMPLEMENTED. {result.notes}"
    if result.status is ModuleStatus.ERROR:
        return f"ERROR. {result.notes}"
    if result.status is ModuleStatus.NO_RESULTS or not result.answer:
        return f"NO RESULTS. {result.notes}"

    lines = [f"Retrieved and verified {len(result.answer.claims)} finding(s)."]
    for claim in result.answer.claims:
        pmids = ", ".join(s.identifier for s in claim.sources)
        lines.append(f"- {claim.text} [{pmids}]")
    if result.notes:
        lines.append(f"\nNotes: {result.notes}")
    lines.append(
        "\nCite these PMIDs in your answer. Do not state anything not listed above."
    )
    return "\n".join(lines)


@beta_tool
def search_literature(query: str, max_results: int = 10, min_year: int | None = None) -> str:
    """Search PubMed and extract verified findings from the abstracts.

    Every finding returned has been checked against its source abstract, so
    anything this returns is safe to cite. Findings that could not be verified
    are discarded before you see them.

    Args:
        query: A PubMed-style search query. Use medical terms, not a natural
            language question. Good: "metformin cardiovascular mortality type 2
            diabetes". Bad: "Does metformin reduce cardiovascular mortality?".
        max_results: How many abstracts to retrieve and analyze (1-25).
        min_year: Only include papers published in or after this year. Use it
            when the question asks about recent evidence.
    """
    # Imported here rather than at module scope to keep the import graph
    # acyclic: the literature package imports schemas, which tools also needs.
    from src.modules.literature import run_literature_module

    # write_summary=False: the orchestrator writes the final answer itself from
    # the rendered claims, so a per-search prose summary would be an extra LLM
    # call whose output is never shown.
    result, _groups = run_literature_module(
        question=query,
        search_query=query,
        max_results=max(1, min(max_results, 25)),
        min_year=min_year,
        write_summary=False,
    )
    _collected.append(result)
    return _render(result)


@beta_tool
def find_repurposing_candidates(disease_or_drug: str, max_candidates: int = 10) -> str:
    """Find drug repurposing candidates via knowledge-graph paths.

    NOT IMPLEMENTED YET. If you call this, report to the user that the
    capability is not built, and do not answer from your own knowledge.

    Args:
        disease_or_drug: The disease to find drugs for, or the drug to find new
            indications for.
        max_candidates: Maximum number of candidates to return.
    """
    result = _find_repurposing(disease_or_drug, max_candidates)
    _collected.append(result)
    return _render(result)


@beta_tool
def match_phenotypes(symptom_description: str, top_k: int = 10) -> str:
    """Rank candidate rare diseases from a free-text symptom description.

    NOT IMPLEMENTED YET. If you call this, report to the user that the
    capability is not built, and do not answer from your own knowledge.

    This is a research triage aid, never a diagnostic tool.

    Args:
        symptom_description: The presenting signs and symptoms, in plain text.
        top_k: How many candidate diagnoses to return.
    """
    result = _match_phenotypes(symptom_description, top_k)
    _collected.append(result)
    return _render(result)


ALL_TOOLS = [search_literature, find_repurposing_candidates, match_phenotypes]
