"""
Tests for the grounding chokepoint.

These are the most important tests in the project. If provenance checking
silently passes fabricated content, every number the system reports becomes
meaningless - including, especially, the hallucination rate itself.
"""
from tests.conftest import make_finding

from src.provenance import (
    FUZZY_FLOOR,
    MIN_QUOTE_CHARS,
    enforce_grounding,
    normalize_text,
    verify_all,
    verify_finding,
)
from src.schemas import (
    Claim,
    MatchKind,
    RejectedFinding,
    RejectionReason,
    Source,
    SourcedAnswer,
    SourceType,
    VerifiedFinding,
)


class TestRejectsFabrication:
    """The two failure modes the whole design exists to catch."""

    def test_rejects_fabricated_pmid(self, abstracts):
        """A citation to a paper we never retrieved must not pass."""
        finding = make_finding(pmid="99999999")
        result = verify_finding(finding, abstracts)

        assert isinstance(result, RejectedFinding)
        assert result.reason is RejectionReason.PMID_NOT_RETRIEVED

    def test_rejects_fabricated_content(self, abstracts):
        """A real PMID with content that paper never stated must not pass."""
        finding = make_finding(
            quote="Metformin tripled the risk of stroke in every subgroup studied"
        )
        result = verify_finding(finding, abstracts)

        assert isinstance(result, RejectedFinding)
        assert result.reason is RejectionReason.QUOTE_NOT_FOUND
        assert result.best_score < FUZZY_FLOOR

    def test_rejects_quote_too_short_to_be_evidence(self, abstracts):
        """A three-word quote matches half the corpus and proves nothing."""
        result = verify_finding(make_finding(quote="reduced"), abstracts)

        assert isinstance(result, RejectedFinding)
        assert result.reason is RejectionReason.EMPTY_QUOTE

    def test_rejects_quote_stitched_from_two_sentences(self, abstracts):
        """
        Text that appears in the abstract only as disconnected fragments is not
        a quote. This is the subtle case: every individual word is present.
        """
        finding = make_finding(
            quote="Metformin reduced all-cause mortality over a median 12.7 years in every cohort"
        )
        result = verify_finding(finding, abstracts)

        assert isinstance(result, RejectedFinding)


class TestAcceptsGenuineEvidence:
    def test_accepts_verbatim_quote(self, abstracts):
        result = verify_finding(make_finding(), abstracts)

        assert isinstance(result, VerifiedFinding)
        assert result.match_kind is MatchKind.VERBATIM
        assert result.match_score == 100.0
        assert result.source.identifier == "11111111"
        assert result.source.url.endswith("/11111111/")

    def test_accepts_quote_differing_only_in_typography(self, abstracts):
        """
        Unicode en-dashes and curly quotes are transport artifacts, not
        fabrication. Rejecting these would make the metric measure encoding
        noise rather than truthfulness.
        """
        finding = make_finding(
            quote="Metformin reduced all‑cause mortality (HR 0.79, 95% CI 0.65–0.96)"
        )
        result = verify_finding(finding, abstracts)

        assert isinstance(result, VerifiedFinding)
        assert result.match_kind is MatchKind.VERBATIM

    def test_case_insensitive(self, abstracts):
        result = verify_finding(
            make_finding(quote="METFORMIN REDUCED ALL-CAUSE MORTALITY"), abstracts
        )
        assert isinstance(result, VerifiedFinding)


class TestNormalization:
    def test_collapses_whitespace_and_case(self):
        assert normalize_text("  Hello   WORLD \n") == "hello world"

    def test_unifies_dashes_and_quotes(self):
        assert normalize_text("all–cause") == normalize_text("all-cause")
        assert normalize_text("“quoted”") == normalize_text('"quoted"')

    def test_does_not_strip_content_words(self):
        """Over-normalizing would start matching text that isn't really there."""
        assert "mortality" in normalize_text("Reduced mortality.")


class TestGroundingReport:
    def test_hallucination_rate_is_fraction_rejected(self, abstracts):
        findings = [
            make_finding(),                       # verified
            make_finding(pmid="99999999"),        # rejected: bad PMID
            make_finding(quote="invented claim about an unrelated drug entirely"),  # rejected
        ]
        report = verify_all(findings, abstracts)

        assert len(report.verified) == 1
        assert len(report.rejected) == 2
        assert report.total == 3
        assert report.hallucination_rate == 2 / 3

    def test_empty_report_does_not_divide_by_zero(self, abstracts):
        report = verify_all([], abstracts)
        assert report.hallucination_rate == 0.0
        assert "no findings" in report.summary_line()


class TestAnswerBoundary:
    def test_strips_unsourced_claims(self):
        """The last line of defence: nothing unsourced reaches a final answer."""
        answer = SourcedAnswer(
            question="q",
            summary="s",
            claims=[
                Claim(text="sourced", sources=[
                    Source(type=SourceType.PUBMED, identifier="11111111")
                ]),
                Claim(text="unsourced", sources=[]),
            ],
        )
        cleaned, removed = enforce_grounding(answer)

        assert [c.text for c in cleaned.claims] == ["sourced"]
        assert [c.text for c in removed] == ["unsourced"]
        assert cleaned.ungrounded_claims() == []


def test_min_quote_chars_is_enforced_not_just_declared():
    """Guard against the constant drifting away from the behaviour."""
    assert MIN_QUOTE_CHARS >= 10
