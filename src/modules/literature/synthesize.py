"""
Verified findings -> a sourced answer, including where the papers disagree.

The important design decision here is that **contradiction detection is
mechanical, not LLM-judged**. Findings are grouped by what they measure, and a
group containing findings that point in opposite directions is flagged by
comparing enum values - not by asking a model whether the evidence "seems
mixed".

That matters because of the failure the project plan warns about: when a model
judges contradiction, you cannot tell genuine disagreement in the literature
from noise in your own extraction. Deriving it from typed fields means a
contradiction is always traceable to two specific findings and two specific
PMIDs that a human can go read.

The LLM writes prose only at the very end, over a fixed set of already-verified
findings. The claims themselves are built from those findings mechanically, so
they are grounded by construction rather than by trusting the prose.
"""
from __future__ import annotations

import re
import time

from rapidfuzz import fuzz

from src.llm import complete_text
from src.provenance import enforce_grounding, normalize_text
from src.schemas import (
    DESIGN_WEIGHT,
    CandidateFinding,
    Claim,
    ConsensusKind,
    EffectDirection,
    EvidenceGroup,
    GroundingReport,
    ModuleResult,
    ModuleStatus,
    SourcedAnswer,
    VerifiedFinding,
)

# Intervention and outcome are compared separately, and both must match for
# two findings to be about "the same thing". Comparing a concatenated
# "intervention + outcome" string instead lets a strong match on one half mask
# a weak match on the other - which is how "aspirin / bleeding" ends up in a
# group with "aspirin / cardiovascular events" and manufactures a
# contradiction that does not exist.
#
# token_set_ratio rather than token_sort_ratio because one phrase is often a
# more specific form of the other ("metformin" vs "metformin therapy"), and
# set-based comparison scores that as the match it plainly is.
_INTERVENTION_FLOOR = 90.0
_OUTCOME_FLOOR = 90.0

_DIRECTIONAL = {EffectDirection.INCREASE, EffectDirection.DECREASE}

_CITATION_RE = re.compile(r"\[(\d{6,9})\]")

SUMMARY_SYSTEM = """\
You write short evidence summaries for a biomedical research tool.

You are given a numbered list of findings that have already been extracted and \
verified against their source papers. Write 3-5 sentences summarizing what \
this body of evidence shows.

Rules:
- State nothing that is not in the findings list. You have no other knowledge \
of this topic.
- Cite every claim with the PMID in square brackets, like [12345678].
- Where the findings disagree, say so plainly and cite both sides. Do not \
smooth over a conflict to produce a tidier summary.
- Where the evidence rests on a single small study, say that.
- Plain prose. No headings, no bullet points, no preamble.
"""


def _same_topic(a: CandidateFinding, b: CandidateFinding) -> bool:
    """Do two findings measure the same intervention against the same outcome?"""
    intervention = fuzz.token_set_ratio(
        normalize_text(a.intervention), normalize_text(b.intervention)
    )
    if intervention < _INTERVENTION_FLOOR:
        return False
    outcome = fuzz.token_set_ratio(normalize_text(a.outcome), normalize_text(b.outcome))
    return outcome >= _OUTCOME_FLOOR


def group_findings(verified: list[VerifiedFinding]) -> list[EvidenceGroup]:
    """
    Cluster findings that measure the same intervention/outcome pair.

    Greedy clustering against each group's first member. Good enough at this
    scale (tens of findings per question) and it keeps grouping deterministic,
    which matters because the eval harness scores contradiction detection
    against the result.

    Known limitation: it matches on surface text, so "metformin" and
    "biguanide therapy" stay in separate groups. Fixing that needs concept
    normalization against a vocabulary like MeSH or RxNorm - a reasonable
    week 6+ improvement once the knowledge graph is loaded anyway.
    """
    buckets: list[list[VerifiedFinding]] = []

    for item in verified:
        for members in buckets:
            if _same_topic(item.finding, members[0].finding):
                members.append(item)
                break
        else:
            buckets.append([item])

    groups: list[EvidenceGroup] = []
    for members in buckets:
        weights: dict[str, float] = {}
        for member in members:
            direction = member.finding.direction
            weights[direction.value] = weights.get(direction.value, 0.0) + DESIGN_WEIGHT.get(
                member.finding.design, 0.3
            )

        present = {m.finding.direction for m in members}
        directional_present = present & _DIRECTIONAL

        if len(directional_present) == 2:
            kind = ConsensusKind.CONTRADICTION
        elif directional_present and EffectDirection.NO_EFFECT in present:
            kind = ConsensusKind.MIXED
        elif len(members) == 1:
            kind = ConsensusKind.SINGLE
        else:
            kind = ConsensusKind.CONSENSUS

        dominant = max(weights, key=weights.get) if weights else EffectDirection.UNCLEAR.value

        groups.append(
            EvidenceGroup(
                intervention=members[0].finding.intervention,
                outcome=members[0].finding.outcome,
                kind=kind,
                findings=members,
                direction_weights=weights,
                dominant_direction=EffectDirection(dominant),
            )
        )

    # Best-supported groups first - most weight of evidence behind them.
    groups.sort(key=lambda g: sum(g.direction_weights.values()), reverse=True)
    return groups


def _claim_text(item: VerifiedFinding) -> str:
    f = item.finding
    verb = {
        EffectDirection.INCREASE: "increased",
        EffectDirection.DECREASE: "decreased",
        EffectDirection.NO_EFFECT: "had no significant effect on",
        EffectDirection.UNCLEAR: "had an unclear effect on",
    }[f.direction]

    text = f"In {f.population}, {f.intervention} {verb} {f.outcome}"
    if f.comparator:
        text += f" compared with {f.comparator}"
    if f.effect_size:
        text += f" ({f.effect_size})"
    design = f.design.value.replace("_", " ")
    text += f" [{design}"
    text += f", n={f.sample_size}]" if f.sample_size else "]"
    return text


def _findings_block(groups: list[EvidenceGroup]) -> str:
    lines: list[str] = []
    index = 1
    for group in groups:
        if group.kind in (ConsensusKind.CONTRADICTION, ConsensusKind.MIXED):
            lines.append(
                f"\n-- {group.kind.value.upper()} on "
                f"{group.intervention} / {group.outcome} --"
            )
        for item in group.findings:
            lines.append(f"{index}. {_claim_text(item)} [{item.source.identifier}]")
            index += 1
    return "\n".join(lines)


def synthesize(
    question: str,
    report: GroundingReport,
    *,
    write_summary: bool = True,
) -> tuple[ModuleResult, list[EvidenceGroup]]:
    """
    Turn a GroundingReport into a ModuleResult with a sourced answer.

    Takes the report rather than raw findings so that it is impossible to call
    this on unverified data - there is no code path from a CandidateFinding to
    a published claim that skips provenance checking.
    """
    started = time.perf_counter()
    usage: dict[str, int] = {}

    if not report.verified:
        return (
            ModuleResult(
                module="literature",
                status=ModuleStatus.NO_RESULTS,
                grounding=report,
                elapsed_seconds=round(time.perf_counter() - started, 3),
                notes=(
                    f"no findings survived verification ({report.summary_line()})"
                    if report.total else "no findings extracted"
                ),
            ),
            [],
        )

    groups = group_findings(report.verified)

    # Claims are built from verified findings, never parsed out of the prose.
    # This is what makes every claim grounded by construction.
    claims = [
        Claim(text=_claim_text(item), sources=[item.source], confidence=item.match_score / 100.0)
        for item in report.verified
    ]

    summary = ""
    notes: list[str] = []
    if write_summary:
        summary, usage = complete_text(
            system=SUMMARY_SYSTEM,
            user=f"Question: {question}\n\nVerified findings:\n{_findings_block(groups)}",
        )

        # The prose is a rendering of already-verified claims, but the model
        # could still cite a PMID that isn't in the set. Cheap to check.
        allowed = {item.source.identifier for item in report.verified}
        invented = set(_CITATION_RE.findall(summary)) - allowed
        if invented:
            notes.append(f"summary cited PMIDs not in the verified set: {sorted(invented)}")

    contradictions = [g for g in groups if g.kind is ConsensusKind.CONTRADICTION]
    if contradictions:
        notes.append(f"{len(contradictions)} contradictory evidence group(s) detected")
    if report.rejected:
        notes.append(report.summary_line())

    answer, removed = enforce_grounding(
        SourcedAnswer(question=question, summary=summary, claims=claims)
    )
    if removed:
        notes.append(f"dropped {len(removed)} ungrounded claim(s) at the answer boundary")

    return (
        ModuleResult(
            module="literature",
            status=ModuleStatus.OK,
            answer=answer,
            grounding=report,
            elapsed_seconds=round(time.perf_counter() - started, 3),
            usage=usage,
            notes="; ".join(notes),
        ),
        groups,
    )
