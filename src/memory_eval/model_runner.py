from __future__ import annotations

from openai import OpenAI


class OpenAIModelRunner:
    def __init__(self, api_key: str, model: str, temperature: float = 0.0) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._temperature = temperature

    def generate_answer(self, prompt: str) -> str:
        response = self._client.responses.create(
            model=self._model,
            input=prompt,
            temperature=self._temperature,
        )
        text = getattr(response, "output_text", None)
        if text:
            return text.strip()
        return str(response).strip()

    def generate_answer_from_messages(self, messages: list[dict[str, str]]) -> str:
        response = self._client.responses.create(
            model=self._model,
            input=messages,
            temperature=self._temperature,
        )
        text = getattr(response, "output_text", None)
        if text:
            return text.strip()
        return str(response).strip()

