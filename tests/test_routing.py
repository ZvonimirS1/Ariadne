"""
Tests that unbuilt modules refuse rather than improvise.

This is the behaviour that makes the stubs worth having now: when asked a
repurposing question, the system must say it cannot answer. The failure mode
being guarded against is a module - or an orchestrator - that fills the gap
with background knowledge and returns something plausible and unsourced.
"""
from src.modules.phenotype import match_phenotypes
from src.modules.repurposing import find_repurposing_candidates
from src.orchestrator.tools import _render, collected, reset_collector
from src.schemas import ModuleStatus


class TestStubsRefuse:
    def test_repurposing_reports_not_implemented(self):
        result = find_repurposing_candidates("amyotrophic lateral sclerosis")

        assert result.status is ModuleStatus.NOT_IMPLEMENTED
        assert result.answer is None

    def test_phenotype_reports_not_implemented(self):
        result = match_phenotypes("ataxia, telangiectasia, recurrent infections")

        assert result.status is ModuleStatus.NOT_IMPLEMENTED
        assert result.answer is None

    def test_stub_output_tells_the_model_not_to_improvise(self):
        """
        The instruction lives in the tool result itself, not only in the system
        prompt - it is the last thing the model reads before answering.
        """
        rendered = _render(find_repurposing_candidates("ALS"))

        assert "NOT IMPLEMENTED" in rendered
        assert "background knowledge" in rendered


class TestCollector:
    def test_reset_clears_previous_run(self):
        reset_collector()
        assert collected() == []

    def test_collected_returns_a_copy(self):
        """Callers must not be able to mutate the collector by accident."""
        reset_collector()
        collected().append("bogus")  # type: ignore[arg-type]
        assert collected() == []
