from __future__ import annotations

import argparse

from memory_eval.config import build_config_from_args
from memory_eval.evaluator import run_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate model memory with MemoryAgentBench.")
    parser.add_argument("--openai-api-key", default=None, help="Optional override for OPENAI_API_KEY.")
    parser.add_argument("--target-model", default="gpt-4.1-mini", help="Target model under test.")
    parser.add_argument("--judge-model", default="gpt-4.1-mini", help="Judge model for scoring.")
    parser.add_argument(
        "--dataset-split",
        default="Accurate_Retrieval",
        help="MemoryAgentBench split (e.g., Accurate_Retrieval, Long_Range_Understanding).",
    )
    parser.add_argument("--max-samples", type=int, default=10, help="Maximum QA pairs to evaluate.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Target model temperature.")
    parser.add_argument("--judge-retries", type=int, default=2, help="Retries for judge JSON parsing.")
    parser.add_argument("--output-path", default="reports/memory_agent_bench_report.json", help="Report file path.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = build_config_from_args(args)
    result = run_evaluation(config)
    print("Evaluation complete")
    print(f"Total questions: {result.total_questions}")
    print(f"Correct questions: {result.correct_questions}")
    print(f"Overall accuracy: {result.overall_accuracy:.4f}")
    print(f"Report: {config.output_path}")


if __name__ == "__main__":
    main()

