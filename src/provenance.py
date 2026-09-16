"""
The grounding chokepoint. Every claim in the system passes through here.

The weak version of provenance is "ask the model to attach a PMID." A model
can invent a PMID, or attach a perfectly real one to a claim that paper never
made. Neither failure is visible downstream, which is exactly why unsourced
LLM output feels trustworthy right up until you check it.

So Ariadne asks for something a model cannot fake without being caught: a
verbatim quote from the abstract. We then check that the quote actually occurs
in the abstract it is attributed to. That catches both failure modes:

  1. Fabricated citation - the PMID isn't in the set we actually retrieved.
  2. Fabricated content  - the PMID is real, but the quote isn't in that paper.

The useful consequence is that the hallucination rate becomes mechanically
measurable, with no human labeling. That number is the project's headline
result, so it needs to come from a check, not a vibe.
"""
from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz

from src.schemas import (
    CandidateFinding,
    Claim,
    GroundingReport,
    MatchKind,
    RejectedFinding,
    RejectionReason,
    Source,
    SourcedAnswer,
    SourceType,
    VerifiedFinding,
)

# An exact match is the happy path. Real abstracts get mangled in transit
# though - Unicode dashes, curly quotes, sectioned abstracts joined with
# newlines - so a near-miss above this score is accepted and labelled
# `paraphrased` rather than being silently passed off as verbatim.
FUZZY_FLOOR = 95.0

# A three-word quote ("risk was reduced") matches half the corpus and proves
# nothing. Below this length we don't treat it as evidence at all.
MIN_QUOTE_CHARS = 12

_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−"), "-")
_SINGLE_QUOTES = dict.fromkeys(map(ord, "‘’‚‛′"), "'")
_DOUBLE_QUOTES = dict.fromkeys(map(ord, "“”„‟″"), '"')
_PUNCT_MAP = {**_DASHES, **_SINGLE_QUOTES, **_DOUBLE_QUOTES}

_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """
    Fold away differences that don't change meaning, so that a quote which is
    genuinely present isn't rejected over a typographic detail.

    Deliberately conservative: it normalizes Unicode punctuation, collapses
    whitespace, and casefolds. It does NOT strip punctuation wholesale or
    stem words - doing so would start matching text that isn't really there,
    which defeats the entire point of the check.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_PUNCT_MAP)
    text = _WHITESPACE.sub(" ", text)
    return text.strip().casefold()


def source_for_pmid(pmid: str, record: dict | None = None) -> Source:
    """Build a checkable Source from a PubMed record. The URL should resolve."""
    return Source(
        type=SourceType.PUBMED,
        identifier=pmid,
        title=(record or {}).get("title") or None,
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
    )


def _haystack(record: dict) -> str:
    """Title plus abstract - a finding may legitimately quote either."""
    return normalize_text(f"{record.get('title') or ''}\n{record.get('abstract') or ''}")


def verify_finding(
    finding: CandidateFinding,
    abstracts: dict[str, dict],
) -> VerifiedFinding | RejectedFinding:
    """
    Check one extracted finding against the documents we actually retrieved.

    `abstracts` is the {pmid: record} mapping straight from the PubMed client.
    Passing the retrieved set (rather than re-fetching by PMID) is what makes
    the first check meaningful: a PMID the model invented simply will not be
    in the set, no matter how real it looks.
    """
    quote = (finding.evidence_quote or "").strip()
    if len(quote) < MIN_QUOTE_CHARS:
        return RejectedFinding(
            finding=finding,
            reason=RejectionReason.EMPTY_QUOTE,
            detail=f"quote is {len(quote)} chars, minimum is {MIN_QUOTE_CHARS}",
        )

    record = abstracts.get(finding.source_pmid)
    if record is None:
        return RejectedFinding(
            finding=finding,
            reason=RejectionReason.PMID_NOT_RETRIEVED,
            detail=f"PMID {finding.source_pmid} was not in the retrieved set",
        )

    haystack = _haystack(record)
    needle = normalize_text(quote)

    if needle in haystack:
        return VerifiedFinding(
            finding=finding,
            source=source_for_pmid(finding.source_pmid, record),
            match_kind=MatchKind.VERBATIM,
            match_score=100.0,
        )

    score = fuzz.partial_ratio(needle, haystack)
    if score >= FUZZY_FLOOR:
        return VerifiedFinding(
            finding=finding,
            source=source_for_pmid(finding.source_pmid, record),
            match_kind=MatchKind.PARAPHRASED,
            match_score=float(score),
        )

    return RejectedFinding(
        finding=finding,
        reason=RejectionReason.QUOTE_NOT_FOUND,
        detail=(
            f"quote not found in PMID {finding.source_pmid} "
            f"(best partial match {score:.1f} < {FUZZY_FLOOR})"
        ),
        best_score=float(score),
    )


def verify_all(
    findings: list[CandidateFinding],
    abstracts: dict[str, dict],
) -> GroundingReport:
    """Run every candidate through verification and tally the result."""
    report = GroundingReport()
    for finding in findings:
        outcome = verify_finding(finding, abstracts)
        if isinstance(outcome, VerifiedFinding):
            report.verified.append(outcome)
        else:
            report.rejected.append(outcome)
    return report


def enforce_grounding(answer: SourcedAnswer) -> tuple[SourcedAnswer, list[Claim]]:
    """
    Last line of defence at the answer boundary: drop any claim that arrived
    without a source at all.

    Nothing should reach this point ungrounded - verify_all() already filtered
    the findings. It exists because "should" is not a guarantee, and a silent
    unsourced claim in a final answer is the one outcome this project is built
    to prevent. Returns the cleaned answer and whatever was removed, so the
    removal gets counted rather than hidden.
    """
    kept = [c for c in answer.claims if c.is_grounded()]
    removed = [c for c in answer.claims if not c.is_grounded()]
    return answer.model_copy(update={"claims": kept}), removed
