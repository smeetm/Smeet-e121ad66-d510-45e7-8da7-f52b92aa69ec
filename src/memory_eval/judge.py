from __future__ import annotations

import json
import re

from openai import OpenAI


JUDGE_SYSTEM_PROMPT = """You are an evaluator for memory QA answers.
Given a question, a ground-truth answer, and a model answer, decide whether the model answer is semantically correct.
Output strict JSON only with this schema:
{"correct": true|false, "reason": "short explanation"}"""


def parse_judge_verdict(raw_text: str) -> tuple[bool, str]:
    raw_text = raw_text.strip()

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if not match:
            raise ValueError("Judge output is not valid JSON")
        data = json.loads(match.group(0))

    if "correct" not in data:
        raise ValueError("Judge output missing 'correct'")
    return bool(data["correct"]), str(data.get("reason", "")).strip()


class OpenAIJudge:
    def __init__(self, api_key: str, model: str, retries: int = 2) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._retries = retries

    def judge(self, question: str, ground_truth: str, model_answer: str) -> tuple[bool, str]:
        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Ground truth answer:\n{ground_truth}\n\n"
            f"Model answer:\n{model_answer}\n\n"
            "Return JSON only."
        )

        last_error: Exception | None = None
        for _ in range(self._retries + 1):
            response = self._client.responses.create(
                model=self._model,
                input=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
            )
            text = getattr(response, "output_text", None) or str(response)
            try:
                return parse_judge_verdict(text)
            except Exception as exc:
                last_error = exc
                continue

        raise ValueError(f"Judge failed to return parseable JSON: {last_error}")

