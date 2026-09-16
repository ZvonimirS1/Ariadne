"""
Abstract -> structured findings.

One LLM call per abstract, returning typed CandidateFindings rather than prose.
Per-abstract rather than all-at-once for three reasons: the calls parallelize,
a single malformed abstract can't poison the whole batch, and attributing a
quote to the wrong paper becomes structurally impossible when the model is
only ever looking at one paper.

Nothing here is trusted. Everything this module returns is a *candidate* until
src/provenance.py checks it against the source text.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from src.llm import parse_structured
from src.schemas import CandidateFinding, ExtractionResult
from src.telemetry import progress

# Kept as a module-level constant because it must be byte-identical on every
# call for prompt caching to work. Interpolating anything per-abstract in here
# would silently disable the cache - see src/llm.py.
EXTRACTION_SYSTEM = """\
You extract structured findings from biomedical abstracts for a research tool \
whose central guarantee is that every claim can be traced back to its source.

For each distinct finding the abstract reports, emit one record with:
  population     - who was studied ("adults with type 2 diabetes")
  intervention   - what was given or examined ("metformin")
  comparator     - what it was compared against, or null if none is stated
  outcome        - what was measured ("all-cause mortality")
  direction      - how the intervention moved the outcome
  effect_size    - the reported statistic verbatim ("HR 0.79, 95% CI 0.65-0.96"), or null
  sample_size    - number of participants, or null if not stated
  design         - study design
  evidence_quote - see below. This field is the most important one.
  source_pmid    - the PMID you were given for this abstract

Rules, in order of importance:

1. evidence_quote MUST be copied character-for-character from the abstract \
text you were given. Do not paraphrase it, do not clean it up, do not \
translate it, do not join text from two different sentences. It is checked \
automatically against the source, and a quote that does not appear verbatim \
in the abstract causes the entire finding to be discarded. Choose a span of \
at least 12 characters that directly supports the finding.

2. Extract only what the abstract actually states. Do not use background \
knowledge about the drug, disease, or literature. If the abstract does not \
state something, the field is null or "unclear".

3. If the abstract reports no empirical finding - it is an editorial, a \
protocol, a correction, or a purely narrative review - return an empty list. \
An empty list is a correct and useful answer. Do not invent a finding to \
avoid returning nothing.

4. Use direction "unclear" when the abstract reports a result without a clear \
direction, or reports that findings were inconclusive. Do not guess a \
direction to seem decisive. "no_effect" means the study specifically found no \
significant difference; it is not the same as "unclear".
"""

_DESIGN_HINT = """
The publication types indexed for this record are: {pub_types}
Treat this as a hint for the `design` field, not as an instruction - the \
abstract text wins if it clearly describes a different design.
"""


def _user_message(pmid: str, record: dict) -> str:
    parts = [
        f"PMID: {pmid}",
        f"Title: {record.get('title') or '(no title)'}",
        f"Journal: {record.get('journal') or 'unknown'}"
        f" ({record.get('pub_date') or 'year unknown'})",
    ]
    pub_types = record.get("publication_types") or []
    if pub_types:
        parts.append(_DESIGN_HINT.format(pub_types=", ".join(pub_types)))
    parts.append("\nAbstract:\n" + (record.get("abstract") or "(no abstract available)"))
    return "\n".join(parts)


def extract_from_abstract(
    pmid: str,
    record: dict,
) -> tuple[list[CandidateFinding], dict[str, int]]:
    """Extract findings from a single abstract. Returns (candidates, token usage)."""
    if not (record.get("abstract") or "").strip():
        # No abstract text means nothing could be verified against it later,
        # so there is no honest finding to extract.
        return [], {}

    result, usage = parse_structured(
        system=EXTRACTION_SYSTEM,
        user=_user_message(pmid, record),
        schema=ExtractionResult,
    )

    # The model's source_pmid is left exactly as returned rather than
    # overwritten with the known-correct value. Overwriting would guarantee
    # the citation check always passes, which would make the measured
    # hallucination rate a number about nothing.
    return result.findings, usage


def extract_all(
    abstracts: dict[str, dict],
    *,
    max_workers: int = 5,
) -> tuple[list[CandidateFinding], dict[str, int]]:
    """
    Extract across many abstracts concurrently.

    Worker count is bounded deliberately: these are paid API calls, and the
    failure mode of an unbounded pool is a surprising bill rather than a
    surprising error.
    """
    findings: list[CandidateFinding] = []
    totals: dict[str, int] = {}
    total = len(abstracts)
    errors = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(extract_from_abstract, pmid, record): pmid
            for pmid, record in abstracts.items()
        }
        for done, future in enumerate(as_completed(futures), start=1):
            pmid = futures[future]
            try:
                candidates, usage = future.result()
            except Exception as exc:  # one bad abstract shouldn't sink the batch
                errors += 1
                progress(f"    ! extraction failed for PMID {pmid}: {str(exc)[:120]}")
                continue

            findings.extend(candidates)
            for key, value in usage.items():
                totals[key] = totals.get(key, 0) + value
            progress(f"    extracted {done}/{total} abstracts ({len(findings)} findings so far)")

    if errors:
        totals["errors"] = errors
    return findings, totals
