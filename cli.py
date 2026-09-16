#!/usr/bin/env python3
"""
Ask Ariadne a question.

    python cli.py "Does metformin reduce cardiovascular mortality in type 2 diabetes?"

The output deliberately separates two things: the model's prose answer, and the
verified claims underneath it. The claims are the real output - each one is
checked against its source abstract - and the grounding line reports how many
extracted findings failed that check. A summary with no claims beneath it is a
red flag, not a clean result.
"""
from __future__ import annotations

import argparse
import sys

from src.config import MissingConfig
from src.orchestrator import answer_question

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask Ariadne a biomedical question.")
    parser.add_argument("question", help="the question to answer")
    parser.add_argument(
        "--max-iterations", type=int, default=8,
        help="cap on orchestrator tool-calling rounds (default: 8)",
    )
    parser.add_argument(
        "--show-claims", type=int, default=10, metavar="N",
        help="how many verified claims to print (display only; default 10, 0 = all)",
    )
    parser.add_argument("--json", action="store_true", help="emit the full result as JSON")
    args = parser.parse_args()

    try:
        result = answer_question(args.question, max_iterations=args.max_iterations)
    except MissingConfig as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(result.model_dump_json(indent=2))
        return 0

    print(f"\n{BOLD}Question{RESET}\n{args.question}\n")
    print(f"{BOLD}Answer{RESET}")
    print(result.answer_text or "(no answer produced)")

    claims = result.all_claims
    if claims:
        shown = claims if args.show_claims == 0 else claims[: args.show_claims]
        print(
            f"\n{BOLD}Verified claims{RESET}  {DIM}(each checked against its source; "
            f"showing {len(shown)} of {len(claims)}){RESET}"
        )
        for claim in shown:
            print(f"  - {claim.text}")
            for source in claim.sources:
                print(f"    {DIM}{source.identifier}  {source.url}{RESET}")
        if len(shown) < len(claims):
            print(f"  {DIM}... {len(claims) - len(shown)} more. "
                  f"Use --show-claims 0 or --json to see all.{RESET}")

    print(f"\n{BOLD}Grounding{RESET}")
    for module_result in result.module_results:
        line = module_result.grounding.summary_line() if module_result.grounding else "n/a"
        print(f"  {module_result.module}: {module_result.status.value} - {line}")
        if module_result.notes:
            print(f"    {DIM}{module_result.notes}{RESET}")

    tokens = result.usage.get("input_tokens", 0) + result.usage.get("output_tokens", 0)
    print(
        f"\n{DIM}{result.iterations} iteration(s), {result.elapsed_seconds}s, "
        f"~{tokens} orchestrator tokens. Run log: data/runs/{result.run_id}.jsonl{RESET}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
