# Free-first BDX-inspired walk policy

This is a beginner-friendly, simulation-only workflow for Open Duck Mini v2. It produces evidence, checkpoints, and an ONNX policy. It does not deploy to a robot, add a head IMU, or control hardware.

## Easiest start on Windows

Double-click START_FREE_TRAINING.cmd in the repository folder. The launcher validates and publishes the branch, reads your fork URL from the existing Git remote, and creates a configured notebook under .local-kaggle. That folder is ignored by Git, so the public notebook and guide stay generic. It then opens Kaggle and highlights the exact file to upload.

Your GitHub username is necessarily visible as the owner of your public fork. Do not put passwords, access tokens, Kaggle credentials, private hardware details, or private artifact data in the notebook or repository.

The launcher intentionally does not use Kaggle CLI upload because that command immediately runs the whole notebook and would bypass the benchmark and human-review gates.

The motion is original and only inspired by visible BDX traits. Do not copy proprietary animation data.

## What is already implemented

- Named-body mass randomization for trunk_assembly and head_assembly.
- Nominal-mass assertions: 0.698526 kg trunk and 0.406607 kg head.
- Moderate and full mass/COM stages.
- Mass and diagonal inertia scaling together.
- Five explicit left/right leg body pairs with a shared scale and at most +/-2% independent mismatch.
- Existing floor friction, actuator friction/armature, joint-offset, and gain randomization remain active.
- The policy observation/action schemas are unchanged.
- No head IMU observations or hardware fields.
- Reproducible seeds, run manifests, checkpoint/ONNX export, and stage timing.
- A 10,000-model randomization audit.
- A 3x3 mass-grid ONNX evaluator.
- A blind five-trait style-review pack and retention checker.
- An original reference-motion generator with a mandatory human review gate.
- A Kaggle-first notebook and a fail-closed paid budget guard.

Exact upstream base commit: b9be205ac64488c23504ca42e5ec790337adeec3

## How limiting is free Kaggle?

Kaggle currently documents 12-hour CPU/GPU notebook sessions, 20 GB of auto-saved /kaggle/working space, and a weekly GPU quota that is generally 30 hours and can vary with demand. P100 and T4 availability is not guaranteed.

Compared with a paid RTX 4090, the free option is mainly limited by:

- slower training;
- a 12-hour session boundary;
- a weekly quota;
- possible interruption or accelerator unavailability;
- the need to download/save artifacts before a session ends.

Those limits are inconvenient, not a reason to skip free testing. The 20M benchmark makes the decision using your real workload.

Official references:

- Kaggle notebooks: https://www.kaggle.com/docs/notebooks
- Kaggle efficient GPU use: https://www.kaggle.com/docs/efficient-gpu-usage
- RunPod RTX 4090: https://www.runpod.io/gpu-models/rtx-4090
- Disney BDX public page: https://la.disneyresearch.com/bdx-droids/
- Reference generator: https://github.com/apirrone/Open_Duck_reference_motion_generator

RunPod lists Community RTX 4090 capacity from roughly US$0.34/hour at the time this guide was written, but availability and live price vary. This project refuses a configured price above US$0.50/hour.

## Step 1: create your GitHub fork

The working branch is codex/free-first-bdx-policy.

1. Open https://github.com/apirrone/Open_Duck_Playground and select Fork.
2. Set your fork as the local repository's origin.
3. Push this branch to your fork.

Generic example:

    git remote add origin https://github.com/YOUR_GITHUB_USER/Open_Duck_Playground.git
    git push -u origin codex/free-first-bdx-policy

The upstream remote should remain:

    https://github.com/apirrone/Open_Duck_Playground.git

When using START_FREE_TRAINING.cmd, do not edit the tracked notebook with your username. The launcher creates a configured, Git-ignored copy for you.

## Step 2: import the notebook into Kaggle

1. Double-click START_FREE_TRAINING.cmd.
2. Wait for two windows:

   - your browser opens Kaggle;
   - File Explorer highlights .local-kaggle/free_first_bdx_walk.ipynb.

3. Sign in to Kaggle if asked.
4. Import the highlighted file:

   - if the Kaggle Code page shows Import Notebook, choose it, choose Local, and select the highlighted file;
   - if a blank notebook editor opens, use File > Import Notebook, then upload the highlighted file.

5. If Kaggle asks how to save it, choose Quick Save. Do not choose Commit & Run or Save & Run All, because this workflow has deliberate stop-and-review points.

You are uploading the .local-kaggle copy, not the similarly named file in the public notebooks folder. The local copy already contains the fork URL, so there is nothing to paste.

## Step 3: enable Internet and the free GPU

With the imported notebook open:

1. Find the Settings pane on the right side of the notebook editor.
2. Under Session options, turn Internet on.
3. Under Accelerator, choose GPU.
4. Choose P100 if Kaggle offers it; otherwise T4 is suitable.
5. Accept the session restart if Kaggle asks.
6. Return to the first code cell and click its triangular Run button.
7. Wait for the final line to say Ready: gpu, show at least one device, and then show the artifacts path.

If Internet or GPU controls are missing or disabled, first check that you are signed in, that Kaggle has completed any requested account/phone verification, and that your weekly GPU quota is not exhausted. After enabling the GPU, rerun the setup cell from the beginning because changing accelerators restarts the session.

The setup cell:

- installs git-lfs and uv;
- clones your fork at codex/free-first-bdx-policy, or updates an existing Kaggle clone to that branch;
- checks UPSTREAM_COMMIT;
- installs dependencies;
- refuses to continue unless JAX reports gpu;
- creates /kaggle/working/artifacts.

If setup reports that UPSTREAM_COMMIT is missing, the older notebook cloned the fork's default branch. Run START_FREE_TRAINING.cmd again, re-import the newly highlighted .local-kaggle notebook, and rerun its setup cell. The corrected setup keeps the existing download but switches it to the required training branch.

If JAX reports Unable to load cuSPARSE and falls back to CPU, Kaggle's system LD_LIBRARY_PATH is shadowing the CUDA libraries bundled with JAX. Re-import the corrected local notebook and rerun setup; it removes that conflicting path before JAX or any training subprocess starts.

Always download the generated ZIP before ending a session.

Official interface reference: https://www.kaggle.com/docs/notebooks

## Step 4: smoke and benchmark

The notebook first runs:

- 1M steps: installation smoke test;
- 20M steps: timed nominal benchmark.

The benchmark is accepted only when:

- evaluation reward rises rather than collapsing;
- at least one checkpoint exists;
- an ONNX export exists;
- MuJoCo/ONNX evaluation loads.

The estimator uses:

    projected 300M time = measured 20M time x 15

Stay free when the projection is under 10 hours. Move to paid resume when any one condition is true:

- projection is 10 hours or more;
- Kaggle interrupts two attempts;
- the weekly quota blocks the selected run.

Do not redo completed stages.

## Step 5: audit the randomizer

The notebook runs:

    uv run python scripts/randomization_audit.py --stage full --samples 10000 --output /kaggle/working/artifacts/randomization_audit.json

It checks:

- all named bodies exist;
- exact configured scale bounds;
- positive, finite mass and inertia;
- per-axis trunk/head COM limits;
- inertia uses the same scale as mass;
- paired-leg mismatch remains bounded.

Do not train the full stage if this audit fails.

## Step 6: robust neutral training

Training adds exactly these steps:

| Stage | Added steps | Cumulative steps | Mass/COM envelope |
|---|---:|---:|---|
| Nominal | 20M | 20M | nominal trunk/head |
| Moderate | 60M | 80M | trunk 90-110%, head 85-115%, COM +/-5 mm |
| Full | 220M | 300M | full configured envelope |

Each stage restores the preceding checkpoint. The stage runner writes stage_result.json with the checkpoint, ONNX path, runtime, seed, and restore source.

Full envelope:

- trunk: 80-130%, about 0.56-0.91 kg;
- head: 65-140%, about 0.26-0.57 kg;
- other links: 95-105%;
- trunk COM: +/-15 mm forward/vertical, +/-10 mm sideways;
- head COM: +/-15 mm forward, +/-10 mm sideways, +/-20 mm vertically.

## Step 7: make the original style reference

Use the notebook reference section or run:

    uv run python scripts/prepare_bdx_reference.py generate --generator-root PATH_TO_GENERATOR --artifact-dir artifacts/bdx_reference

It generates eight motions covering standing, forward/backward, sideways, and turning commands with:

- walk_com_height 0.19;
- walk_foot_height 0.045;
- walk_trunk_pitch -4 degrees;
- single_support_duration 0.18;
- feet_spacing 0.176.

Replay each generated JSON locally:

    uv run scripts/replay_motion.py -f MOTION.json --hardware

Look for:

1. waddle;
2. slight crouch;
3. light bounce;
4. deliberate foot lift;
5. stable upper-body timing.

Record approval only after inspection:

    uv run python scripts/prepare_bdx_reference.py approve --generator-root PATH_TO_GENERATOR --artifact-dir artifacts/bdx_reference --review-note "Replayed all eight motions and checked all five traits."

Fit and copy only after approval:

    uv run python scripts/prepare_bdx_reference.py fit --generator-root PATH_TO_GENERATOR --artifact-dir artifacts/bdx_reference --playground-data playground/open_duck_mini_v2/data

The fit command refuses to run without the approval record.

## Step 8: style fine-tuning

1. Start seeds 201, 202, and 203 from the accepted 300M neutral checkpoint.
2. Train each for 30M steps with full randomization.
3. Evaluate each candidate.
4. Make blind A/B videos against the neutral policy.
5. Pick the best stable candidate.
6. Add 120M steps only to that candidate, giving 150M style steps.

Keep imitation_reward_weight_scale at 1.0. If a candidate remains stable but visibly ignores the reference, retry only that control at 1.5. Do not change other rewards first.

## Step 9: objective evaluation

Neutral or style mass grid:

    uv run python scripts/evaluate_mass_grid.py --onnx POLICY.onnx --episodes 20 --seconds 20 --output artifacts/POLICY_mass_grid.json

The evaluator uses:

- light/nominal/heavy trunk;
- light/nominal/heavy head;
- 20 randomized episodes per cell;
- 20 seconds per episode;
- a 0.10 m/s forward command;
- the same full COM and other-link envelope.

Acceptance:

- non-corner cells: at least 18/20 survive;
- extreme corners: at least 16/20 survive;
- mean speed error for survivors: no more than 15%.

Blind review pack:

    uv run python scripts/make_blind_style_review.py --neutral-video neutral.mp4 --style-video style.mp4 --output-dir artifacts/blind_review

Have someone fill review_form.json without opening answer_key.json.

Style retention check:

    uv run python scripts/check_style_acceptance.py --neutral-report artifacts/neutral_mass_grid.json --style-report artifacts/style_mass_grid.json --review-form artifacts/blind_review/review_form.json --answer-key artifacts/blind_review/answer_key.json --output artifacts/style_acceptance.json

Style passes when:

- survival retention is at least 90%;
- tracking retention is at least 90%;
- the style policy wins at least three of five visible traits.

Export smoke:

    uv run python scripts/evaluate_mass_grid.py --onnx POLICY.onnx --episodes 1 --seconds 60 --output artifacts/export_60s.json

That evaluates all nine mass cells for 60 seconds, which is stricter than the required nominal, heavy-head/light-body, and heavy-body/heavy-head subset.

## Step 10: paid resume only when triggered

Check the live RunPod price before creating a pod. Use Community RTX 4090 only at US$0.50/hour or less.

Before each paid segment:

    uv run python scripts/paid_budget_guard.py --rate LIVE_RATE --elapsed-hours USED_SO_FAR --planned-hours NEXT_SEGMENT

Paid rules:

- upload the last Kaggle artifact ZIP;
- clone the same fork and verify the same upstream commit;
- restore the existing checkpoint;
- stop working spend at US$8;
- reserve US$2 for export/recovery;
- download artifacts;
- terminate the pod immediately.

The scripts never create or terminate cloud resources automatically.

## Failure recovery

If training produces NaNs or reward collapse:

1. Stop the failed stage.
2. Keep the last good checkpoint.
3. Halve trunk/head COM limits for the retry.

       uv run python scripts/run_training_stage.py ... --com-offset-scale 0.5

4. Resume from the last good checkpoint.
5. If it fails again, use moderate mass ranges.
6. Change rewards only after those two recovery attempts.

stage_failure.json records the same recovery rule.

## Artifact checklist

Keep every item below under the session artifact directory:

- run_manifest.json;
- stage_result.json;
- checkpoints;
- ONNX exports;
- TensorBoard logs;
- compute_decision.json;
- randomization_audit.json;
- reference recordings and approval;
- polynomial_coefficients.pkl;
- mass-grid reports;
- blind-review form and answer key;
- style_acceptance.json;
- evaluation videos;
- workspace/fork/upstream metadata;
- seeds.

Hardware variants outside the trained mass/COM envelope require new simulation validation and retraining.
