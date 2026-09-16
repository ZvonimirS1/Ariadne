"""
Shared test fixtures.

Every test in this suite runs offline and costs nothing: no PubMed calls, no
LLM calls. That is deliberate - a test suite you hesitate to run because it
costs money is a test suite you stop running.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.schemas import CandidateFinding, EffectDirection, StudyDesign  # noqa: E402


@pytest.fixture
def abstracts() -> dict[str, dict]:
    """A small retrieved-document set standing in for a PubMed response."""
    return {
        "11111111": {
            "title": "Metformin and mortality in type 2 diabetes: a randomized trial",
            "abstract": (
                "RESULTS: Metformin reduced all-cause mortality "
                "(HR 0.79, 95% CI 0.65-0.96) compared with placebo in 4,209 "
                "patients with type 2 diabetes over a median 5.3 years."
            ),
            "journal": "Journal of Testing",
            "pub_date": "2021",
            "publication_types": ["Randomized Controlled Trial"],
        },
        "22222222": {
            "title": "Metformin in older adults: a cohort study",
            "abstract": (
                "RESULTS: In this cohort of 1,200 older adults, metformin use was "
                "associated with increased all-cause mortality (HR 1.21, 95% CI 1.04-1.41)."
            ),
            "journal": "Journal of Testing",
            "pub_date": "2022",
            "publication_types": ["Observational Study"],
        },
    }


def make_finding(
    pmid: str = "11111111",
    quote: str = "Metformin reduced all-cause mortality",
    direction: EffectDirection = EffectDirection.DECREASE,
    design: StudyDesign = StudyDesign.RCT,
    intervention: str = "metformin",
    outcome: str = "all-cause mortality",
) -> CandidateFinding:
    return CandidateFinding(
        population="adults with type 2 diabetes",
        intervention=intervention,
        comparator="placebo",
        outcome=outcome,
        direction=direction,
        design=design,
        evidence_quote=quote,
        source_pmid=pmid,
    )


@pytest.fixture
def finding_factory():
    return make_finding
