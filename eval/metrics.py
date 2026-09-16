"""
Scoring for Module 1.

Four numbers matter, and they measure different failures:

  citation verification rate - did extracted findings trace to their source?
  hallucination rate         - the inverse. The project's headline metric.
  direction accuracy         - was the extracted effect direction correct?
  contradiction F1           - did it spot real disagreement, and only real
                               disagreement?

Direction accuracy and hallucination rate are deliberately separate. A system
can be perfectly grounded and still wrong - quoting a real sentence but reading
the effect backwards - and that failure is invisible if you only track
grounding.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from src.schemas import ConsensusKind, EffectDirection, EvidenceGroup, ModuleResult, ModuleStatus

# Claude Opus 5, USD per million tokens. Used for reporting run cost.
_INPUT_COST_PER_MTOK = 5.00
_OUTPUT_COST_PER_MTOK = 25.00
_CACHE_READ_COST_PER_MTOK = 0.50


class QuestionScore(BaseModel):
    id: str
    status: str
    expected_direction: str
    predicted_direction: str | None = None
    direction_correct: bool = False
    expect_contradiction: bool = False
    contradiction_detected: bool = False
    n_verified: int = 0
    n_rejected: int = 0
    hallucination_rate: float = 0.0
    retrieval_recall: float | None = None  # None when no hand-verified PMIDs
    elapsed_seconds: float = 0.0
    usage: dict[str, int] = Field(default_factory=dict)
    notes: str = ""


class EvalReport(BaseModel):
    module: str
    scores: list[QuestionScore] = Field(default_factory=list)

    # --- aggregates ---

    @property
    def answered(self) -> list[QuestionScore]:
        """Questions that actually produced findings. Others can't be scored on direction."""
        return [s for s in self.scores if s.status == ModuleStatus.OK.value]

    @property
    def direction_accuracy(self) -> float | None:
        scored = self.answered
        if not scored:
            return None
        return sum(s.direction_correct for s in scored) / len(scored)

    @property
    def total_verified(self) -> int:
        return sum(s.n_verified for s in self.scores)

    @property
    def total_rejected(self) -> int:
        return sum(s.n_rejected for s in self.scores)

    @property
    def hallucination_rate(self) -> float:
        """Pooled across all findings, not averaged per question - questions
        extract different numbers of findings, and averaging rates would let a
        question with two findings outweigh one with twenty."""
        total = self.total_verified + self.total_rejected
        return self.total_rejected / total if total else 0.0

    @property
    def citation_verification_rate(self) -> float:
        return 1.0 - self.hallucination_rate

    def contradiction_prf(self) -> tuple[float, float, float]:
        """Precision, recall, F1 for detecting genuine disagreement."""
        tp = sum(s.contradiction_detected and s.expect_contradiction for s in self.answered)
        fp = sum(s.contradiction_detected and not s.expect_contradiction for s in self.answered)
        fn = sum(not s.contradiction_detected and s.expect_contradiction for s in self.answered)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return precision, recall, f1

    @property
    def usage_totals(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for score in self.scores:
            for key, value in score.usage.items():
                totals[key] = totals.get(key, 0) + value
        return totals

    @property
    def estimated_cost_usd(self) -> float:
        u = self.usage_totals
        return (
            u.get("input_tokens", 0) / 1e6 * _INPUT_COST_PER_MTOK
            + u.get("output_tokens", 0) / 1e6 * _OUTPUT_COST_PER_MTOK
            + u.get("cache_read_input_tokens", 0) / 1e6 * _CACHE_READ_COST_PER_MTOK
        )


def _predicted_direction(groups: list[EvidenceGroup]) -> EffectDirection | None:
    """
    The best-supported group's direction.

    group_findings() already sorts by total evidence weight, so the first group
    is the one with the most support behind it.
    """
    return groups[0].dominant_direction if groups else None


def score_question(
    spec: dict,
    result: ModuleResult,
    groups: list[EvidenceGroup],
) -> QuestionScore:
    expected = spec["expected_direction"]
    predicted = _predicted_direction(groups)
    grounding = result.grounding

    known_pmids = set(spec.get("known_pmids") or [])
    retrieval_recall: float | None = None
    if known_pmids and grounding:
        found = {v.source.identifier for v in grounding.verified}
        retrieval_recall = len(known_pmids & found) / len(known_pmids)

    return QuestionScore(
        id=spec["id"],
        status=result.status.value,
        expected_direction=expected,
        predicted_direction=predicted.value if predicted else None,
        direction_correct=bool(predicted and predicted.value == expected),
        expect_contradiction=bool(spec.get("expect_contradiction")),
        contradiction_detected=any(g.kind is ConsensusKind.CONTRADICTION for g in groups),
        n_verified=len(grounding.verified) if grounding else 0,
        n_rejected=len(grounding.rejected) if grounding else 0,
        hallucination_rate=grounding.hallucination_rate if grounding else 0.0,
        retrieval_recall=retrieval_recall,
        elapsed_seconds=result.elapsed_seconds,
        usage=result.usage,
        notes=result.notes,
    )


def format_report(report: EvalReport) -> str:
    """A plain-text table. These numbers go straight into the writeup."""
    lines = [
        "",
        f"  {'question':<28} {'status':<10} {'dir':<9} {'ok':<4} {'verif':<6} {'rej':<5} {'halluc':<7}",
        f"  {'-' * 78}",
    ]
    for s in report.scores:
        mark = "yes" if s.direction_correct else "no"
        lines.append(
            f"  {s.id:<28} {s.status:<10} {str(s.predicted_direction or '-'):<9} "
            f"{mark:<4} {s.n_verified:<6} {s.n_rejected:<5} {s.hallucination_rate:>6.1%}"
        )

    precision, recall, f1 = report.contradiction_prf()
    accuracy = report.direction_accuracy
    recalls = [s.retrieval_recall for s in report.scores if s.retrieval_recall is not None]

    lines += [
        f"  {'-' * 78}",
        "",
        f"  questions scored          {len(report.answered)} / {len(report.scores)}",
        f"  direction accuracy        {accuracy:.1%}" if accuracy is not None
        else "  direction accuracy        n/a (nothing scored)",
        f"  citation verification     {report.citation_verification_rate:.1%} "
        f"({report.total_verified} verified / "
        f"{report.total_verified + report.total_rejected} extracted)",
        f"  HALLUCINATION RATE        {report.hallucination_rate:.1%}",
        f"  contradiction P/R/F1      {precision:.2f} / {recall:.2f} / {f1:.2f}",
    ]
    if recalls:
        lines.append(f"  retrieval recall          {sum(recalls) / len(recalls):.1%} "
                     f"({len(recalls)} question(s) with verified PMIDs)")
    else:
        lines.append("  retrieval recall          not measured "
                     "(no hand-verified PMIDs in dataset)")
    lines += [
        f"  estimated cost            ${report.estimated_cost_usd:.2f}",
        "",
    ]
    return "\n".join(lines)
