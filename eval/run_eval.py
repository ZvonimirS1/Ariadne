#!/usr/bin/env python3
"""
Run the evaluation harness.

    python eval/run_eval.py --module literature
    python eval/run_eval.py --module literature --limit 2   # quick check

This costs real money - one LLM call per retrieved abstract. Use --limit while
iterating. PubMed responses are cached on disk, so reruns re-extract but do not
re-download.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.harness import run_literature_eval  # noqa: E402
from eval.metrics import format_report  # noqa: E402
from src.config import MissingConfig  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Ariadne's modules.")
    parser.add_argument("--module", default="literature", choices=["literature"],
                        help="which module to evaluate (2 and 3 are not built yet)")
    parser.add_argument("--limit", type=int, default=None,
                        help="only run the first N questions")
    parser.add_argument("--max-results", type=int, default=10,
                        help="abstracts retrieved per question (default: 10)")
    parser.add_argument("--with-summary", action="store_true",
                        help="also generate prose summaries (costs more, scores nothing)")
    parser.add_argument("--out", type=Path, default=None,
                        help="write the full report as JSON here")
    args = parser.parse_args()

    print(f"\nEvaluating: {args.module}")
    try:
        report = run_literature_eval(
            limit=args.limit,
            max_results=args.max_results,
            write_summary=args.with_summary,
        )
    except MissingConfig as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"{exc}", file=sys.stderr)
        return 2

    print(format_report(report))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = json.loads(report.model_dump_json())
        payload["aggregates"] = {
            "direction_accuracy": report.direction_accuracy,
            "hallucination_rate": report.hallucination_rate,
            "citation_verification_rate": report.citation_verification_rate,
            "contradiction_prf": report.contradiction_prf(),
            "estimated_cost_usd": report.estimated_cost_usd,
        }
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  wrote {args.out}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
