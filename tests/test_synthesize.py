"""
Tests for contradiction detection and grouping.

The property under test throughout: disagreement between papers is derived
from typed fields, so it is reproducible and traceable to specific PMIDs -
never a model's opinion about whether evidence "seems mixed".
"""
from tests.conftest import make_finding

from src.modules.literature.synthesize import group_findings, synthesize
from src.provenance import verify_all
from src.schemas import ConsensusKind, EffectDirection, ModuleStatus, StudyDesign


def _verified(abstracts, findings):
    report = verify_all(findings, abstracts)
    assert not report.rejected, "fixture findings should all verify"
    return report


class TestContradictionDetection:
    def test_opposite_directions_are_a_contradiction(self, abstracts):
        report = _verified(abstracts, [
            make_finding(pmid="11111111", quote="Metformin reduced all-cause mortality",
                         direction=EffectDirection.DECREASE),
            make_finding(pmid="22222222",
                         quote="metformin use was associated with increased all-cause mortality",
                         direction=EffectDirection.INCREASE, design=StudyDesign.COHORT),
        ])
        groups = group_findings(report.verified)

        assert len(groups) == 1
        assert groups[0].kind is ConsensusKind.CONTRADICTION

    def test_contradiction_is_traceable_to_both_papers(self, abstracts):
        """A flagged contradiction must point at papers a human can go read."""
        report = _verified(abstracts, [
            make_finding(pmid="11111111", quote="Metformin reduced all-cause mortality",
                         direction=EffectDirection.DECREASE),
            make_finding(pmid="22222222",
                         quote="metformin use was associated with increased all-cause mortality",
                         direction=EffectDirection.INCREASE, design=StudyDesign.COHORT),
        ])
        group = group_findings(report.verified)[0]

        assert {f.source.identifier for f in group.findings} == {"11111111", "22222222"}

    def test_design_weighting_favours_the_rct_over_the_cohort(self, abstracts):
        """A cohort study should not outweigh an RCT just by disagreeing with it."""
        report = _verified(abstracts, [
            make_finding(pmid="11111111", quote="Metformin reduced all-cause mortality",
                         direction=EffectDirection.DECREASE, design=StudyDesign.RCT),
            make_finding(pmid="22222222",
                         quote="metformin use was associated with increased all-cause mortality",
                         direction=EffectDirection.INCREASE, design=StudyDesign.COHORT),
        ])
        group = group_findings(report.verified)[0]

        assert group.dominant_direction is EffectDirection.DECREASE
        assert group.direction_weights["decrease"] > group.direction_weights["increase"]

    def test_agreement_is_consensus_not_contradiction(self, abstracts):
        report = _verified(abstracts, [
            make_finding(pmid="11111111", quote="Metformin reduced all-cause mortality",
                         direction=EffectDirection.DECREASE),
            make_finding(pmid="22222222", quote="associated with increased all-cause mortality",
                         direction=EffectDirection.DECREASE, design=StudyDesign.COHORT),
        ])
        assert group_findings(report.verified)[0].kind is ConsensusKind.CONSENSUS

    def test_single_finding_is_not_consensus(self, abstracts):
        """One paper agreeing with itself is not agreement."""
        report = _verified(abstracts, [make_finding()])
        assert group_findings(report.verified)[0].kind is ConsensusKind.SINGLE


class TestGrouping:
    def test_different_outcomes_do_not_group(self, abstracts):
        """Unrelated outcomes must never be merged into a false contradiction."""
        report = _verified(abstracts, [
            make_finding(pmid="11111111", quote="Metformin reduced all-cause mortality",
                         outcome="all-cause mortality", direction=EffectDirection.DECREASE),
            make_finding(pmid="22222222", quote="associated with increased all-cause mortality",
                         outcome="risk of bleeding", intervention="aspirin",
                         direction=EffectDirection.INCREASE),
        ])
        assert len(group_findings(report.verified)) == 2

    def test_phrasing_variants_of_the_same_intervention_group_together(self, abstracts):
        """'metformin' and 'metformin therapy' describe the same thing."""
        report = _verified(abstracts, [
            make_finding(pmid="11111111", quote="Metformin reduced all-cause mortality",
                         intervention="metformin"),
            make_finding(pmid="22222222", quote="associated with increased all-cause mortality",
                         intervention="metformin therapy", design=StudyDesign.COHORT),
        ])
        assert len(group_findings(report.verified)) == 1


class TestSynthesizeContract:
    def test_no_verified_findings_yields_no_results_not_an_answer(self, abstracts):
        """
        With nothing verified, the correct output is "no results" - never a
        summary written from the model's own knowledge.
        """
        report = verify_all([make_finding(pmid="99999999")], abstracts)
        result, groups = synthesize("does metformin help?", report, write_summary=False)

        assert result.status is ModuleStatus.NO_RESULTS
        assert result.answer is None
        assert groups == []

    def test_every_claim_carries_a_source(self, abstracts):
        """Claims are built from verified findings, so this holds by construction."""
        report = _verified(abstracts, [make_finding()])
        result, _ = synthesize("does metformin help?", report, write_summary=False)

        assert result.status is ModuleStatus.OK
        assert result.answer.claims
        assert all(c.is_grounded() for c in result.answer.claims)
        assert result.answer.ungrounded_claims() == []

    def test_rejected_findings_are_reported_not_hidden(self, abstracts):
        report = verify_all([make_finding(), make_finding(pmid="99999999")], abstracts)
        result, _ = synthesize("q", report, write_summary=False)

        assert result.grounding.hallucination_rate == 0.5
        assert "hallucination rate" in result.notes
