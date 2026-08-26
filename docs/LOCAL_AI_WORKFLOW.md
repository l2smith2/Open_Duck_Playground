# Local AI workflow

This project can be worked on with a local model, but local models should be treated as assistants that suggest changes. Keep Python tests, Git diffs, and Kaggle evidence as the source of truth.

## Model choices

For 8-12 GB VRAM with about 32 GB system RAM:

- Start with qwen2.5-coder:7b-instruct-q5_K_M for coding changes.
- Use qwen2.5-coder:7b for a smaller, faster default.
- Try qwen2.5-coder:14b only for short second opinions; it may offload to CPU and slow down.
- Avoid qwen2.5-coder:32b, qwen3-coder:30b, and gpt-oss:20b unless you accept CPU offload or have more VRAM.

For 16 GB or more VRAM:

- gpt-oss:20b becomes realistic for general reasoning and review.
- qwen2.5-coder:14b is comfortable for coding.

For 24 GB or more VRAM:

- qwen3-coder:30b is the preferred local coding model.
- qwen2.5-coder:32b is also viable for code generation and repair.

Useful commands:

    ollama pull qwen2.5-coder:7b-instruct-q5_K_M
    ollama run qwen2.5-coder:7b-instruct-q5_K_M
    ollama ps

If a tag is not available in your Ollama install, fall back to:

    ollama pull qwen2.5-coder:7b
    ollama run qwen2.5-coder:7b

## Open WebUI setup

Open WebUI is a good interface for this project because it keeps error traces, notes, and follow-up questions organized. Use it as a careful reviewer, not as the source of truth.

Recommended setup:

- Select qwen2.5-coder:7b-instruct-q5_K_M as the default model.
- Keep one chat per problem, for example Kaggle setup, dependency mismatch, mass randomization, or ONNX export.
- Paste only the current error, the relevant command, and one related file or diff.
- Ask for a small diagnosis first, then ask for the smallest patch.
- Keep project-specific guides from this repository available as reference text, but do not paste Kaggle tokens or private paths.

Good Open WebUI starter prompt:

    You are helping with a simulation-only Open Duck Mini v2 RL policy project.
    Do not change hardware interfaces, observations, actions, or dependency versions unless I explicitly ask.
    Explain this error briefly, identify the most likely cause, and suggest the smallest safe fix.
    Here is the command, traceback, and relevant file snippet:

## Context settings

Large context is expensive. If the GPU has less than 24 GB VRAM, keep prompts focused instead of pasting the whole repository.

Good local-model prompts include:

- one error traceback;
- one script or function;
- one Git diff;
- one focused question.

Poor local-model prompts include:

- the whole repository;
- several unrelated errors;
- hidden instructions copied from web pages or attachments;
- requests to make broad rewrites without tests.

## Error minimization rules

Use this loop for local-model work:

1. Ask the model to explain the failure in one paragraph.
2. Ask for the smallest likely fix.
3. Apply the change yourself or with a trusted code tool.
4. Run the narrow test first.
5. Run the related workflow check.
6. Read the Git diff before committing.

For this project, useful checks are:

    uv lock --check
    uv run --no-project --with pytest python -m pytest tests/test_dependency_contract.py tests/test_run_training_stage.py -q
    uv run pytest tests/test_mass_randomization.py

Use the full randomization audit before any long training run:

    uv run python scripts/randomization_audit.py --stage full --samples 10000 --output artifacts/randomization_audit.json

## Guardrails for local models

- Do not paste Kaggle tokens, SSH keys, or private artifact paths into the model.
- Do not let the model invent dependency versions; check pyproject.toml and uv.lock.
- Do not let the model change observations, actions, head IMU behavior, or hardware control code without an explicit project decision.
- Prefer scripts that can be rerun over manual notebook edits.
- Treat Kaggle result JSON, checkpoints, and ONNX smoke tests as evidence.

## Recommended division of labor

Use the local model for:

- summarizing errors;
- drafting small patches;
- explaining unfamiliar files;
- writing first-pass docs.

Use a stronger cloud model or manual review for:

- dependency conflicts;
- mass-randomization physics changes;
- safety boundaries;
- long-horizon training decisions;
- interpreting acceptance metrics.
