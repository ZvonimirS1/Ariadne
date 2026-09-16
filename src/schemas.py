"""
Shared data models for every module in the system.

The one rule that matters more than anything else in this project:
no Claim exists without a Source. If a module wants to assert something,
it has to attach where that assertion came from. This is what makes the
system's output checkable instead of just plausible-sounding.

Every module (literature synthesis, drug repurposing, rare disease
matching) should return its results using these shapes, or subclasses
of them, so the orchestrator and the eval harness can treat all three
modules the same way.

The type split that carries the most weight here is CandidateFinding vs.
VerifiedFinding. The model produces Candidates; only `src/provenance.py`
can turn one into a Verified. Nothing downstream accepts a Candidate, so
unchecked model output is structurally prevented from reaching an answer.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    PUBMED = "pubmed"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    HPO_ORPHANET = "hpo_orphanet"
    MIMIC_IV = "mimic_iv"


class Source(BaseModel):
    """A single, checkable pointer back to where a claim came from."""
    type: SourceType
    identifier: str  # e.g. a PMID, a graph edge ID, an HPO term ID
    title: str | None = None
    url: str | None = None


class Claim(BaseModel):
    """One atomic, sourced assertion."""
    text: str
    sources: list[Source] = Field(default_factory=list)
    confidence: float | None = None  # 0-1, optional, module-defined

    def is_grounded(self) -> bool:
        """A claim with zero sources should never make it into a final answer."""
        return len(self.sources) > 0


class SourcedAnswer(BaseModel):
    """The standard return shape for any module's top-level response."""
    question: str
    summary: str
    claims: list[Claim] = Field(default_factory=list)

    def ungrounded_claims(self) -> list[Claim]:
        """Handy for the eval harness: anything that slipped through unsourced."""
        return [c for c in self.claims if not c.is_grounded()]


# --------------------------------------------------------------------------
# Literature extraction
# --------------------------------------------------------------------------

class EffectDirection(str, Enum):
    """
    Which way the intervention moved the outcome.

    `unclear` is a first-class answer, not a failure. Forcing a direction out
    of an abstract that doesn't state one is how extraction noise gets
    mistaken for genuine disagreement between papers later on.
    """
    INCREASE = "increase"
    DECREASE = "decrease"
    NO_EFFECT = "no_effect"
    UNCLEAR = "unclear"


class StudyDesign(str, Enum):
    META_ANALYSIS = "meta_analysis"
    RCT = "rct"
    COHORT = "cohort"
    CASE_CONTROL = "case_control"
    CASE_REPORT = "case_report"
    REVIEW = "review"
    OTHER = "other"


# Used when papers disagree, so a case report doesn't get to "contradict" a
# meta-analysis on equal footing. Higher wins.
DESIGN_WEIGHT: dict[StudyDesign, float] = {
    StudyDesign.META_ANALYSIS: 1.0,
    StudyDesign.RCT: 0.9,
    StudyDesign.COHORT: 0.6,
    StudyDesign.CASE_CONTROL: 0.5,
    StudyDesign.REVIEW: 0.4,
    StudyDesign.CASE_REPORT: 0.2,
    StudyDesign.OTHER: 0.3,
}


class CandidateFinding(BaseModel):
    """
    One finding as the model extracted it - NOT yet checked against the source.

    `evidence_quote` is the load-bearing field: it must be copied verbatim out
    of the abstract. src/provenance.py verifies that the quote genuinely occurs
    in the abstract it is attributed to, which is what lets us measure a
    hallucination rate without hand-labeling anything.
    """
    population: str
    intervention: str
    comparator: str | None = None
    outcome: str
    direction: EffectDirection
    effect_size: str | None = None  # free text: "HR 0.79 (95% CI 0.65-0.96)"
    sample_size: int | None = None
    design: StudyDesign
    evidence_quote: str
    source_pmid: str


class ExtractionResult(BaseModel):
    """Root model for one abstract's extraction call (structured output needs one)."""
    findings: list[CandidateFinding] = Field(default_factory=list)


class MatchKind(str, Enum):
    """How closely the evidence quote matched the source text."""
    VERBATIM = "verbatim"      # exact substring after normalization
    PARAPHRASED = "paraphrased"  # cleared the fuzzy floor but isn't exact


class VerifiedFinding(BaseModel):
    """A CandidateFinding that survived provenance checking. Only these count."""
    finding: CandidateFinding
    source: Source
    match_kind: MatchKind
    match_score: float  # 100.0 for an exact match


class ConsensusKind(str, Enum):
    """What a set of findings about the same intervention/outcome pair adds up to."""
    SINGLE = "single"                # only one finding - no agreement to assess
    CONSENSUS = "consensus"          # all findings point the same way
    MIXED = "mixed"                  # a directional finding alongside "no effect"
    CONTRADICTION = "contradiction"  # findings point in opposite directions


class EvidenceGroup(BaseModel):
    """
    Findings that measure the same thing, grouped so disagreement is visible.

    `direction_weights` sums study-design weights per direction, so "one case
    report says otherwise" doesn't read as an even split against a
    meta-analysis. Kept as a typed object rather than prose because the eval
    harness scores contradiction detection against it.
    """
    intervention: str
    outcome: str
    kind: ConsensusKind
    findings: list[VerifiedFinding] = Field(default_factory=list)
    direction_weights: dict[str, float] = Field(default_factory=dict)
    dominant_direction: EffectDirection


class RejectionReason(str, Enum):
    EMPTY_QUOTE = "empty_quote"
    PMID_NOT_RETRIEVED = "pmid_not_retrieved"  # cited a paper we never fetched
    QUOTE_NOT_FOUND = "quote_not_found"        # real paper, invented content


class RejectedFinding(BaseModel):
    finding: CandidateFinding
    reason: RejectionReason
    detail: str = ""
    best_score: float = 0.0  # best fuzzy score seen, for tuning the threshold


class GroundingReport(BaseModel):
    """
    The headline metric of the whole project.

    Rejected findings are kept rather than dropped - their rate IS the
    measurement, so throwing them away would discard the result.
    """
    verified: list[VerifiedFinding] = Field(default_factory=list)
    rejected: list[RejectedFinding] = Field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.verified) + len(self.rejected)

    @property
    def hallucination_rate(self) -> float:
        """Fraction of extracted findings that could not be traced to their source."""
        return len(self.rejected) / self.total if self.total else 0.0

    def summary_line(self) -> str:
        if not self.total:
            return "no findings extracted"
        return (
            f"verified {len(self.verified)} / rejected {len(self.rejected)} "
            f"-> hallucination rate {self.hallucination_rate:.1%}"
        )


# --------------------------------------------------------------------------
# Module envelope
# --------------------------------------------------------------------------

class ModuleStatus(str, Enum):
    OK = "ok"
    NO_RESULTS = "no_results"
    NOT_IMPLEMENTED = "not_implemented"
    ERROR = "error"


class ModuleResult(BaseModel):
    """
    Uniform envelope every module returns, so the orchestrator and the eval
    harness can treat all three the same way regardless of data source.
    """
    module: str
    status: ModuleStatus
    answer: SourcedAnswer | None = None
    grounding: GroundingReport | None = None
    elapsed_seconds: float = 0.0
    usage: dict[str, int] = Field(default_factory=dict)
    notes: str = ""
