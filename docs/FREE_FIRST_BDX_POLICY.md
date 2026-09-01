# Free-first BDX-inspired walk policy

This is a beginner-friendly, simulation-only workflow for Open Duck Mini v2. It produces evidence, checkpoints, and an ONNX policy. It does not deploy to a robot, add a head IMU, or control hardware.

## Easiest start on Windows

Double-click START_FREE_TRAINING.cmd in the repository folder. The launcher validates and publishes the branch, reads your fork URL from the existing Git remote, and creates a configured notebook under .local-kaggle. That folder is ignored by Git, so the public notebook and guide stay generic. It then opens Kaggle and highlights the exact file to upload.

Your GitHub username is necessarily visible as the owner of your public fork. Do not put passwords, access tokens, Kaggle credentials, private hardware details, or private artifact data in the notebook or repository.

The launcher intentionally does not use Kaggle CLI upload because that command immediately runs the whole notebook and would bypass the benchmark and human-review gates.

After the Kaggle API is configured, START_KAGGLE_BATCH.cmd can run a private non-interactive smoke plus benchmark job and download its outputs. Use that path for repeatable runs where browser persistence is not needed. See docs/KAGGLE_API_BATCH.md.

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

## Step 3: enable Internet, persistence, and the free GPU

With the imported notebook open:

1. Find the Settings pane on the right side of the notebook editor.
2. Under Session options, turn Internet on.
3. Turn File persistence on so /kaggle/working can survive interactive restarts.
4. Under Accelerator, choose T4 x2; P100 remains acceptable when available.
5. Accept the session restart if Kaggle asks.
6. Confirm File persistence is still on after the restart.
7. Return to the first code cell and click its triangular Run button.
8. Wait for the final line to say Ready: gpu, show at least one device, and then show the artifacts path.

If Internet or GPU controls are missing or disabled, first check that you are signed in, that Kaggle has completed any requested account/phone verification, and that your weekly GPU quota is not exhausted. After enabling the GPU, rerun the setup cell from the beginning because changing accelerators restarts the session.

The setup cell:

- installs git-lfs and uv;
- clones your fork at codex/free-first-bdx-policy, or updates an existing Kaggle clone to that branch;
- checks UPSTREAM_COMMIT;
- installs dependencies, including the compatible playground 0.0.5 release;
- uses the committed lockfile for Brax 0.13.0, JAX/JAXlib 0.6.2, and MuJoCo/MJX 3.3.3;
- verifies all required MuJoCo Playground imports before training;
- allows the first import to download mujoco_menagerie and ignores its status messages when reading the package version;
- refuses to continue unless JAX reports gpu;
- creates /kaggle/working/artifacts.

If setup reports that UPSTREAM_COMMIT is missing, the older notebook cloned the fork's default branch. Run START_FREE_TRAINING.cmd again, re-import the newly highlighted .local-kaggle notebook, and rerun its setup cell. The corrected setup keeps the existing download but switches it to the required training branch.

Do not use Kaggle's Dependency Manager for this project. The notebook's uv sync command owns the reproducible Python environment.

If uv sync reports Failed to query Python interpreter for .venv/bin/python with Permission denied, Kaggle File persistence restored a broken virtual environment. Re-import the corrected notebook and run Setup again. Setup recreates only .venv; the repository, artifacts, and checkpoints are preserved.

If JAX reports Unable to load cuSPARSE or CUDA error 303 and falls back to CPU, re-import the corrected local notebook and rerun setup. It first proves that Kaggle attached a GPU, then keeps only Kaggle's mounted NVIDIA driver path while preventing the system CUDA toolkit from shadowing JAX's bundled libraries.

If the new setup stops with No NVIDIA GPU is attached, choose T4 x2 or another GPU in Settings, accept Kaggle's session restart, and run Setup again. Dependency changes cannot fix a CPU-only session.

If training reports that mujoco_playground._src.collision is missing, an unpinned setup installed playground 0.2.0. Re-import the corrected notebook and rerun Setup; uv will downgrade it to the compatible 0.0.5 release before the smoke test.

The failed smoke directory can be reused. A successful retry automatically removes its stale stage_failure.json marker, so no manual Kaggle file cleanup is required.

If PPO reports that jax.device_put_replicated is deprecated, an unlocked setup selected a JAX release that is too new for this Brax trainer. Re-import the corrected notebook and rerun Setup. It uses the committed lockfile, downgrades the virtual environment to JAX/JAXlib 0.6.2 and Brax 0.13.0, and verifies those versions before training.

Always download the generated ZIP before ending a session.

Official interface reference: https://www.kaggle.com/docs/notebooks

## Resuming after an interrupted session

Kaggle file persistence is best effort, and a fresh notebook import always
starts with an empty /kaggle/working. To make a session recoverable, upload
the last artifact ZIP as a private Kaggle Dataset:

    kaggle datasets init -p FOLDER_CONTAINING_THE_ZIP
    # set title and id in dataset-metadata.json, then:
    kaggle datasets create -p FOLDER_CONTAINING_THE_ZIP

Attach it to the notebook with Add Input, then run Setup followed by the
restore/status cell. It calls:

    uv run python scripts/restore_artifacts.py --artifacts /kaggle/working/artifacts
    uv run python scripts/pipeline_status.py --artifacts /kaggle/working/artifacts

restore_artifacts.py finds the artifact ZIP among the attached inputs and moves
in anything missing. It never overwrites artifacts that are already present, so
it is safe to run every session, and it does nothing when no backup is attached.

pipeline_status.py reads the artifact directory and reports which stages are
genuinely complete, the state of the reference motion, and what to run next.
Notebook Python state never survives a restart even when files do, so use this
rather than relying on memory of what ran last session. It also runs locally
against a downloaded bundle:

    uv run python scripts/pipeline_status.py --artifacts kaggle-outputs/YOUR_BUNDLE

## Step 4: smoke and benchmark

The notebook first runs:

- 1M steps: installation smoke test;
- 20M steps: timed nominal benchmark.

If a session ends after a stage prints a complete stage_result.json, rerunning the same notebook section reuses that stage when its configuration, checkpoint, and ONNX export all match. A partial or mismatched stage runs again.

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

Each stage restores the preceding checkpoint. The stage runner writes stage_result.json with the checkpoint, ONNX path, runtime, seed, restore source, and a fingerprint of the reward configuration.

A stage is only reusable if it trained against the same objective, so changing a reward weight or tracking sigma changes that fingerprint and the stage runs again rather than being silently reused. The tracking-reward correction in Step 7b changed it, so checkpoints produced before that correction will retrain from the beginning of this sequence. That is deliberate: they were optimised for a reward that paid more for holding a heading than for going anywhere.

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

- walk_com_height 0.21;
- walk_foot_height 0.04;
- walk_trunk_pitch -6 degrees;
- single_support_duration 0.225;
- feet_spacing 0.16.

These come from configs/bdx_inspired_reference.json, which is the only place
to change them; the notebook and prepare_bdx_reference.py both read it, so
there is nothing else to edit.

Before spending GPU on a candidate reference, prove it rewards walking more
than marching in place:

    uv run python scripts/check_reference_locomotion_incentive.py \
        --reference artifacts/bdx_reference/polynomial_coefficients.pkl \
        --walking-onnx NEUTRAL.onnx --marching-onnx COLLAPSED_STYLE.onnx \
        --output artifacts/bdx_reference/locomotion_incentive.json

The imitation reward is dominated by joint_pos, an unbounded quadratic on joint
angles, plus a foot-contact match. Both are satisfied by cycling the legs on the
spot. If a reference's fore-aft leg swing is small next to its lateral sway,
marching in place scores higher than walking, and 30M steps of fine-tuning
reliably find that optimum with no gradient path out. Two style attempts failed
exactly this way, each surviving every mass-grid cell at 0.002 m/s against a
0.10 m/s command, so mass-grid survival does not detect it.

Measured against the 220M neutral policy and a collapsed style seed, the stock
reference scores +1.62 on this check and the two failed bdx bundles score -0.78
and -1.67. Clearing the +0.25 threshold is the minimum, not the goal: a
reference near zero is still far weaker than the one that trains a walking
policy, and every point it gives away has to be made up elsewhere.

walk_foot_height is the strongest lever on fore-aft swing; lowering
walk_com_height also helps. Do not read "deliberate foot lift" as a reason to
reduce walk_foot_height: halving it to 0.02 is what flattened the leg swing.
It is now back to 0.04, the generator's own default before that halving, as
the best-motivated first candidate rather than an arbitrary new number.

Testing a candidate value does not need Cell 7's human-review gate or a
finished bundle: `prepare_bdx_reference.py screen` fits whatever `generate`
just recorded straight to a small screening_coefficients.pkl, skipping
approval entirely, so it cannot reach training:

    uv run python scripts/prepare_bdx_reference.py generate --generator-root PATH_TO_GENERATOR --artifact-dir artifacts/bdx_reference
    uv run python scripts/prepare_bdx_reference.py screen   --generator-root PATH_TO_GENERATOR --artifact-dir artifacts/bdx_reference

Both generate and screen are CPU-only kinematics/fitting work, not GPU
training, so they are cheap to repeat while searching for a value. Score the
resulting screening_coefficients.pkl with check_reward_locomotion_incentive.py
wherever you already have a walking and a collapsed ONNX cached -- that does
not have to be the same machine that ran generate/screen. Only run the full
generate -> replay -> approve -> fit -> install sequence once a candidate
clears the gate.

single_support_duration is not a style dial. It sets the stride period exactly,
as period = 2.4 * single_support_duration, and 0.225 is the value that keeps the
0.540 s period the neutral checkpoint is trained on.

A fine-tune cannot retime a gait. The imitation reward's joint_pos term is an
unbounded quadratic on joint angles, so a policy whose stride cannot phase-lock
to the reference clock scores strictly worse than one that stops walking and
sits near the reference mean pose. Swapping in a faster reference does not speed
the gait up, it collapses it: style seeds 201, 202, and 203 were first trained
against a 0.432 s reference and all three marched in place at 0.002 m/s against
a 0.10 m/s command, surviving every mass-grid cell while going nowhere.

Do not read placo_defaults.json as the baseline. Its single_support_duration of
0.17 is not the cadence of the shipped reference the neutral stages train
against; treating it as such, and nudging to 0.18, is what cut the period by
20%. install_reference_motion.py now refuses to install a reference whose stride
period differs from the installed one unless --allow-cadence-change says
training restarts from scratch.

These values are tuned to keep both knees inside the model's joint range and
bending in the natural direction (see the joint_limit_note in
configs/bdx_inspired_reference.json). The upstream gait generator disables
its own IK joint limits, and its solver has been observed to land on a
different local optimum for a byte-identical preset depending on the machine
it runs on -- almost certainly floating-point non-associativity (differing
CPU/thread counts changing summation order) tipping a near-boundary solution
into an invalid branch. A configuration proven safe on one machine is
therefore not proven safe on another.

Because of that, generate() checks every generated motion on whichever machine
actually generated it, and retries with a small walk_com_height nudge, up to 5
times, before failing loudly. This runs automatically as part of the command
above; a persistent failure means the style parameters need a real change,
not a retry, and the error message says so.

The check refuses broken solutions, not merely unreachable ones. A reference is
a soft imitation target, never a trajectory the robot replays, so it does not
have to fit inside the model's joint ranges -- and the stock reference that
produces a walking policy does not: all 240 of its entries exceed the knee
range, peaking near +2.01 rad against a +-1.571 limit. Holding style references
inside the range capped their leg swing at roughly half the stock reference's
and let marching in place outscore walking. So the check refuses two things: a
knee that bends backwards, which is the IK solver landing on its inverted
branch, and any joint more than 0.5 rad outside its range. Over-flexion in the
natural direction is allowed, exactly as the working reference does it.

Replay each generated JSON locally to check the style, not the joint limits
(those are already checked automatically):

    uv run python scripts/replay_bdx_reference.py -f MOTION.json

The upstream generator's own scripts/replay_motion.py depends on
FramesViewer/placo, which ships no Windows wheels and hits a GLX threading
error under WSLg, so it does not work on this platform. This fork's
replay_bdx_reference.py plays the same recorded JSON back in the MuJoCo
viewer this project already depends on, on any platform that can run
mujoco_infer.py. -f also accepts a directory with --check, to manually
re-verify an existing bundle without opening a viewer window:

    uv run python scripts/replay_bdx_reference.py -f RECORDINGS_DIR --check

That also prints a bundle fingerprint: a content hash of the exact motions you
are about to watch. Keep it; approval below is bound to it.

Look for:

1. waddle;
2. slight crouch;
3. light bounce;
4. deliberate foot lift;
5. stable upper-body timing.

Record approval only after inspection, passing the fingerprint the check
printed for the bundle you actually watched:

    uv run python scripts/prepare_bdx_reference.py approve --generator-root PATH_TO_GENERATOR --artifact-dir artifacts/bdx_reference --review-note "Replayed all eight motions and checked all five traits." --expect-fingerprint FINGERPRINT

Because the upstream solver can produce different motions from identical input
on a different machine, approve refuses a bundle whose fingerprint does not
match: a regenerated bundle cannot inherit the approval of one you reviewed.
For the same reason the notebook keeps generation and approval in separate
cells, so rerunning the approval step can never regenerate the motions.

Fit and copy only after approval:

    uv run python scripts/prepare_bdx_reference.py fit --generator-root PATH_TO_GENERATOR --artifact-dir artifacts/bdx_reference --playground-data playground/open_duck_mini_v2/data

The fit command refuses to run without the approval record.

Motions are filed under the velocity they actually realise, measured from each
recording's root trajectory, because the generator does not honour the
requested dx/dy/dtheta. To repair a bundle fitted before that was true, without
regenerating and losing its approval:

    uv run python scripts/prepare_bdx_reference.py rekey --artifact-dir artifacts/bdx_reference

Re-install the reference afterwards, and retrain every stage that used the old
keys: a policy trained against a mislabelled reference is not recoverable by
fine-tuning.

## Step 7b: check the reward configuration, not just the reference

A reference and a reward configuration can each make marching in place the
better-paid behaviour, and only the reference had a gate. This one command
checks both: it scores every term in the reward for a policy that walks and a
policy that collapsed, and prints the imitation breakdown -- the reference
check -- beside the total.

    uv run python scripts/check_reward_locomotion_incentive.py         --reference artifacts/bdx_reference/polynomial_coefficients.pkl         --bundle kaggle-outputs/YOUR_BUNDLE         --cache artifacts/reward_incentive_rollouts.json         --output artifacts/reward_incentive.json

--bundle picks both policies out of an artifact bundle, so you never paste a
checkpoint path: the walking one from the furthest-trained neutral stage, the
collapsed one from the first style seed. Use --walking-onnx/--marching-onnx if
you need specific exports.

Rolling the policies out is the slow part and does not depend on any setting, so
--cache records the raw per-step inputs and everything else is scored from it
instantly -- weights, both tracking sigmas, and the imitation betas:

    uv run python scripts/check_reward_locomotion_incentive.py         --reference artifacts/bdx_reference/polynomial_coefficients.pkl         --cache artifacts/reward_incentive_rollouts.json         --scales alive=2.0 --scales feet_air_time=3.0 --ang-tracking-sigma 0.5

Run it before changing any reward weight. Two 75-minute training runs were spent
on plausible-sounding changes that this would have refused in a minute.

Four results worth knowing before you tune anything:

- The weights inherited from upstream scored walking only +0.09 ahead of
  standing still at a 0.10 m/s command, even under the stock reference that
  trains a working policy. Two thirds of that shortfall was tracking_ang_vel
  sharing its sigma with tracking_lin_vel: walking swings the torso in yaw at
  about 0.13 rad/s, which the shared sigma read as a large error, so standing
  still collected about 2.4 reward per step more than walking. The current
  configuration separates them and scores +3.6.
- alive cannot cause or cure this. Both behaviours survive, so both collect it
  in full; changing it moves the margin's share of the per-step total but never
  its sign.
- The unwired foot rewards in playground/common/rewards.py do not help either.
  The collapsed policy never leaves the ground, so every term gated on first
  contact is zero for it; feet_air_time at weight 3.0 moves the margin by
  +0.009, and wiring all five in at the weights other projects use makes it
  slightly worse.
- Rescaling the imitation reward's betas does nothing. Sweeping the linear beta
  from 8 to 200 moves the stock margin by at most +0.05 and makes the bdx bundle
  monotonically worse. The exponentials were never saturated: they see the
  instantaneous velocity error, which oscillates.

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

Push recovery, standing and walking:

    uv run python scripts/evaluate_push_recovery.py --onnx POLICY.onnx --output artifacts/POLICY_push_recovery.json

Training disturbs the floating base every 5-10 s with an instantaneous linear
x/y velocity delta of 0.1-1.0 m/s in a random horizontal direction, and 10% of
sampled commands are all-zero, so standing recovery is trained too. This
evaluator reproduces that disturbance model exactly and sweeps past the trained
range, reporting for each condition the largest magnitude every episode
survived and the magnitude where survival first falls below half.

It is diagnostic, not a gate: it exits non-zero only when --require-magnitude
is given. Rotational and toppling disturbances are out of distribution and are
deliberately not scored.

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
