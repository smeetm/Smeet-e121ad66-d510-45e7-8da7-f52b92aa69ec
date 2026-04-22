from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class EvalConfig:
    openai_api_key: str
    target_model: str
    judge_model: str
    dataset_split: str
    max_samples: int | None
    output_path: Path
    temperature: float
    judge_retries: int


def build_config_from_args(args: object) -> EvalConfig:
    # Keep local project runs predictable: values in .env override stale shell vars.
    # CLI flag still has highest priority.
    load_dotenv(override=True)

    openai_api_key = getattr(args, "openai_api_key", None) or os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is required. Pass --openai-api-key or set environment variable.")

    return EvalConfig(
        openai_api_key=openai_api_key,
        target_model=getattr(args, "target_model"),
        judge_model=getattr(args, "judge_model"),
        dataset_split=getattr(args, "dataset_split"),
        max_samples=getattr(args, "max_samples"),
        output_path=Path(getattr(args, "output_path")),
        temperature=float(getattr(args, "temperature")),
        judge_retries=int(getattr(args, "judge_retries")),
    )

