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
