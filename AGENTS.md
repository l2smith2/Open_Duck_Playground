# AGENTS.md

## Purpose

This fork develops a reproducible, simulation-only Open Duck Mini v2 locomotion
policy with robust trunk/head mass randomization and an original BDX-inspired
walking style.

The first deliverable is simulation evidence plus an ONNX policy. Do not add
physical robot deployment, motor control, cameras, microphones, ROS, a head
IMU, or other hardware interfaces without a separate explicit request.

## Living-document rule

Keep this file current. In the same commit as a meaningful workflow change,
update the relevant section here when any of these change:

- setup commands, dependencies, branches, or cloud environments;
- notebook stages, checkpoints, artifact locations, or recovery procedures;
- mass/COM randomization limits or policy interfaces;
- tests, acceptance thresholds, safety boundaries, or known setup fixes.

Also update README.md and docs/FREE_FIRST_BDX_POLICY.md when user-facing
instructions change. Do not record temporary debugging output or personal data.

## Public-repository privacy

- Keep tracked guides and notebooks generic for all users.
- Never commit usernames, access tokens, API keys, passwords, Kaggle
  credentials, private hardware details, or personal filesystem paths.
- The tracked notebook uses YOUR_GITHUB_USER.
- START_FREE_TRAINING.cmd derives the fork URL from Git and writes the
  configured notebook only to .local-kaggle/, which must remain ignored.
- Store training artifacts outside Git. Never commit checkpoints, ONNX exports,
  videos, logs, or generated credential files.

## Repository and reproducibility

- Work on codex/free-first-bdx-policy.
- Preserve the upstream remote as
  https://github.com/apirrone/Open_Duck_Playground.git.
- UPSTREAM_COMMIT records the exact upstream base. Do not change it unless the
  fork is intentionally rebased and the full workflow is revalidated.
- Use named MuJoCo bodies, never numeric body IDs.
- Keep seeds, configuration, timing, checkpoints, exports, logs, and evaluation
  reports in the artifact bundle.

## Main entry points

- Beginner launcher: START_FREE_TRAINING.cmd
- Kaggle template: notebooks/free_first_bdx_walk.ipynb
- Full guide: docs/FREE_FIRST_BDX_POLICY.md
- Training stage: scripts/run_training_stage.py
- Randomization audit: scripts/randomization_audit.py
- Mass-grid evaluation: scripts/evaluate_mass_grid.py
- Style review: scripts/make_blind_style_review.py
- Paid guard: scripts/paid_budget_guard.py

Kaggle artifacts belong under /kaggle/working/artifacts. Always package and
download them before the session ends.

## Kaggle setup invariants

- Clone or fetch codex/free-first-bdx-policy, not the fork's default branch.
- Confirm UPSTREAM_COMMIT before dependency installation or training.
- Require nvidia-smi -L to report at least one attached GPU before installing.
- JAX's pip wheels provide the CUDA toolkit libraries. Keep only Kaggle's
  /usr/local/nvidia driver directories in LD_LIBRARY_PATH so the driver remains
  visible without system CUDA libraries shadowing bundled cuSPARSE.
- Do not use Kaggle's Dependency Manager; uv sync owns project dependencies.
- Use UV_LINK_MODE=copy on Kaggle to avoid unsupported hardlinks.
- Setup passes only when jax.default_backend() is gpu.
- Record jax.local_device_count(). T4 x2 should normally report two devices;
  P100 reports one.
- Keep Kaggle File persistence on to preserve /kaggle/working across interactive
  restarts, but treat it as best-effort and continue packaging/downloading
  artifact ZIPs before a session ends.
- Do not use Save & Run All while developing because the notebook contains
  benchmark and human-review gates.

## Training sequence

1. Run the 1M smoke test.
2. Run the 20M timed nominal benchmark.
3. Continue cumulatively to 300M neutral steps through nominal, moderate, and
   full randomization. Resume checkpoints; do not restart completed stages.
4. Generate and visually approve the original BDX-inspired reference.
5. Train three 30M style seeds, select one using objective evaluation and blind
   review, then extend only the winner to 150M style steps.
6. Use paid compute only when the benchmark or Kaggle limits trigger it, with
   the documented US$10 cap.

## Policy and randomization boundaries

- Keep observations and actions unchanged; mass and COM values remain hidden.
- Do not add head-IMU observations or hardware fields.
- Scale inertia with mass and reject non-positive or non-finite values.
- Preserve existing friction, actuator, backlash, joint-offset, and gain
  randomization.
- Preserve paired-leg symmetry and the configured mismatch bound.

## Verification before handoff

- Parse the notebook as JSON and compile every Python code cell.
- Parse scripts/start_free_training.ps1 with the PowerShell parser.
- Run git diff --check.
- Run affected unit tests; for randomization changes, include
  uv run pytest tests/test_mass_randomization.py.
- Run the 10,000-model audit before full training.
- Confirm the configured .local-kaggle notebook is ignored.
- Scan tracked files for credential-like strings and personal identifiers.
- Keep the working tree clean and report the published commit.

## Failure recovery

On NaNs or reward collapse, resume the last good checkpoint with COM offsets
halved. If failure repeats, return to moderate mass ranges before changing
rewards.
