# Memory Evaluation MVP

Python CLI for running MemoryAgentBench evaluation against a configurable OpenAI target model and scoring with an OpenAI LLM judge.

## What it does

- Loads MemoryAgentBench from Hugging Face (`ai-hyz/MemoryAgentBench`)
- Feeds each document from the `context` column to the model sequentially
- Asks each question from the `questions` array after all documents are fed
- Compares against the aligned answer in `answers[question_index]`
- Gets answers from a target model
- Scores each answer with LLM-as-Judge
- Computes overall accuracy and writes a JSON report

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set your API key:

```bash
set OPENAI_API_KEY=your_key_here
```

Optional (recommended for IDE run/debug):

- Copy `.env.example` to `.env`
- Set `OPENAI_API_KEY` in `.env`
- Use the included debug config `Memory Eval: Run CLI Module`
- Or use `Python: Current File (Project Env)` to run a file with project import paths preconfigured

## Run a smoke test

```bash
set PYTHONPATH=src
python -m memory_eval.cli --max-samples 10 --output-path reports/smoke.json
```

## Main options

- `--target-model`: model under test (default: `gpt-4.1-mini`)
- `--judge-model`: judge model (default: `gpt-4.1-mini`)
- `--dataset-split`: MemoryAgentBench split (default: `Accurate_Retrieval`)
- `--max-samples`: number of QA pairs to evaluate
- `--judge-retries`: retries when judge output is malformed
- `--output-path`: report JSON file path

## Report output

The report JSON includes:

- run metadata and model config
- `result.total_questions`
- `result.correct_questions`
- `result.overall_accuracy`
- per-sample records with question index, answers, model output, and judge verdict

## Notes

- v1 computes overall accuracy only.
- MemoryAgentBench answers can contain multiple acceptable variants; evaluator passes all variants to the judge.

