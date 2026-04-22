from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json

from tqdm import tqdm

from memory_eval.config import EvalConfig
from memory_eval.dataset_loader import EvalSample, load_memory_agent_bench_samples
from memory_eval.judge import OpenAIJudge
from memory_eval.model_runner import OpenAIModelRunner


@dataclass(frozen=True)
class EvalResult:
    total_questions: int
    correct_questions: int
    overall_accuracy: float


def _build_document_feed_messages(sample: EvalSample) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are participating in a memory benchmark. "
                "You will receive a sequence of documents. "
                "Memorize details across all documents and answer questions accurately."
            ),
        },
    ]

    for idx, doc in enumerate(sample.documents, start=1):
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Document {idx}:\n{doc}\n\n"
                    "Acknowledge with: OK"
                ),
            }
        )
        messages.append({"role": "assistant", "content": "OK"})

    messages.append(
        {
            "role": "user",
            "content": f"Question:\n{sample.question}\n\nAnswer concisely and directly.",
        }
    )
    return messages


def run_evaluation(config: EvalConfig) -> EvalResult:
    runner = OpenAIModelRunner(
        api_key=config.openai_api_key,
        model=config.target_model,
        temperature=config.temperature,
    )
    judge = OpenAIJudge(
        api_key=config.openai_api_key,
        model=config.judge_model,
        retries=config.judge_retries,
    )

    total = 0
    correct = 0
    records: list[dict] = []
    samples = list(load_memory_agent_bench_samples(config.dataset_split, config.max_samples))
    for sample in tqdm(samples, desc="Evaluating", unit="qa"):
        messages = _build_document_feed_messages(sample)
        model_answer = runner.generate_answer_from_messages(messages)
        ground_truth_text = " | ".join(sample.ground_truth_answers)
        is_correct, reason = judge.judge(sample.question, ground_truth_text, model_answer)

        total += 1
        if is_correct:
            correct += 1

        records.append(
            {
                "row_id": sample.row_id,
                "split": sample.split,
                "question_index": sample.question_index,
                "source": sample.source,
                "question": sample.question,
                "ground_truth_answers": sample.ground_truth_answers,
                "model_answer": model_answer,
                "correct": is_correct,
                "judge_reason": reason,
                "document_count": len(sample.documents),
            }
        )

    accuracy = (correct / total) if total else 0.0
    result = EvalResult(total_questions=total, correct_questions=correct, overall_accuracy=accuracy)

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "target_model": config.target_model,
            "judge_model": config.judge_model,
            "dataset": "ai-hyz/MemoryAgentBench",
            "dataset_split": config.dataset_split,
            "max_samples": config.max_samples,
        },
        "result": asdict(result),
        "records": records,
    }

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return result

