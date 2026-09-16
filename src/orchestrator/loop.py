"""
The tool-calling loop.

Uses the SDK's tool runner rather than a hand-written while loop - it handles
dispatch and the message bookkeeping, and `max_iterations` gives the runaway
guard for free. All tools here are client-side, so the `pause_turn` restart
handling needed for server-side tools does not apply.

What this returns matters: alongside the model's prose it returns the
structured ModuleResults collected during the run. Callers should treat the
typed claims as authoritative and the prose as presentation.
"""
from __future__ import annotations

import time

from pydantic import BaseModel, Field

from src import config
from src.llm import get_client
from src.orchestrator.router import ORCHESTRATOR_SYSTEM
from src.orchestrator.tools import ALL_TOOLS, collected, reset_collector
from src.schemas import ModuleResult
from src.telemetry import RunLog, progress


class OrchestratorResult(BaseModel):
    question: str
    answer_text: str
    module_results: list[ModuleResult] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)
    elapsed_seconds: float = 0.0
    iterations: int = 0
    run_id: str = ""

    @property
    def all_claims(self):
        """Every verified claim gathered this run - the authoritative output."""
        return [
            claim
            for result in self.module_results
            if result.answer
            for claim in result.answer.claims
        ]


def answer_question(
    question: str,
    *,
    max_iterations: int = 8,
    log: RunLog | None = None,
) -> OrchestratorResult:
    """Route a question through the modules and return a sourced answer."""
    started = time.perf_counter()
    owns_log = log is None
    log = log or RunLog("orchestrator")
    log.event("question", text=question)

    reset_collector()
    progress("planning which tools to use...")

    runner = get_client().beta.messages.tool_runner(
        model=config.ANTHROPIC_MODEL,
        max_tokens=8000,
        system=ORCHESTRATOR_SYSTEM,
        tools=ALL_TOOLS,
        messages=[{"role": "user", "content": question}],
        max_iterations=max_iterations,
    )

    last_message = None
    iterations = 0
    usage: dict[str, int] = {}

    for message in runner:
        iterations += 1
        last_message = message

        message_usage = getattr(message, "usage", None)
        if message_usage is not None:
            for key in (
                "input_tokens",
                "output_tokens",
                "cache_read_input_tokens",
                "cache_creation_input_tokens",
            ):
                usage[key] = usage.get(key, 0) + (getattr(message_usage, key, 0) or 0)

        tool_names = [b.name for b in message.content if b.type == "tool_use"]
        if tool_names:
            log.event("tool_calls", tools=tool_names, iteration=iterations)
            progress(f"step {iterations}: calling {', '.join(tool_names)}")
        else:
            progress("writing final answer...")

    answer_text = ""
    if last_message is not None:
        if getattr(last_message, "stop_reason", None) == "refusal":
            answer_text = "The model declined to answer this question."
        else:
            answer_text = "".join(
                b.text for b in last_message.content if b.type == "text"
            ).strip()

    module_results = collected()
    log.add_usage(usage)
    for result in module_results:
        log.add_usage(result.usage)
        log.event(
            "module_result",
            module=result.module,
            status=result.status.value,
            claims=len(result.answer.claims) if result.answer else 0,
            hallucination_rate=(
                round(result.grounding.hallucination_rate, 4) if result.grounding else None
            ),
            notes=result.notes,
        )

    result = OrchestratorResult(
        question=question,
        answer_text=answer_text,
        module_results=module_results,
        usage=usage,
        elapsed_seconds=round(time.perf_counter() - started, 3),
        iterations=iterations,
        run_id=log.run_id,
    )

    if owns_log:
        log.close(ok=True, iterations=iterations)
    return result
