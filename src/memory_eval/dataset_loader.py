from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import re

from datasets import load_dataset


@dataclass(frozen=True)
class EvalSample:
    row_id: str
    split: str
    question_index: int
    question: str
    ground_truth_answers: list[str]
    documents: list[str]
    source: str | None


_DOC_SPLIT_PATTERN = re.compile(r"(?=Document\s+\d+\s*:)", flags=re.IGNORECASE)


def _split_context_documents(context: str) -> list[str]:
    context = (context or "").strip()
    if not context:
        return []
    chunks = [chunk.strip() for chunk in _DOC_SPLIT_PATTERN.split(context) if chunk.strip()]
    if chunks:
        return chunks
    return [context]


def _normalize_answers(raw_answer: object) -> list[str]:
    if isinstance(raw_answer, list):
        return [str(item).strip() for item in raw_answer if str(item).strip()]
    text = str(raw_answer).strip()
    return [text] if text else []


def load_memory_agent_bench_samples(split: str, max_samples: int | None) -> Iterable[EvalSample]:
    if max_samples is not None and max_samples <= 0:
        return

    dataset = load_dataset("ai-hyz/MemoryAgentBench", "default", split=split)
    emitted = 0

    for row_idx, row in enumerate(dataset):
        documents = _split_context_documents(str(row.get("context", "")))
        questions = row.get("questions", []) or []
        answers = row.get("answers", []) or []
        metadata = row.get("metadata", {}) or {}
        source = metadata.get("source")

        total_pairs = min(len(questions), len(answers))
        for question_index in range(total_pairs):
            question = str(questions[question_index]).strip()
            ground_truth_answers = _normalize_answers(answers[question_index])
            if not question or not ground_truth_answers:
                continue
            yield EvalSample(
                row_id=f"{split}:{row_idx}",
                split=split,
                question_index=question_index,
                question=question,
                ground_truth_answers=ground_truth_answers,
                documents=documents,
                source=str(source) if source is not None else None,
            )
            emitted += 1

            if max_samples is not None and emitted >= max_samples:
                return

