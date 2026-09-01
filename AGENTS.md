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

Also update README.md and docs/FREE_FIRST_BDX_POLICY.md whenever user-facing
API, launcher, or local-AI workflow instructions change. Do not record temporary
debugging output or personal data.

## Public-repository privacy

- Keep tracked guides and notebooks generic for all users.
- Never commit usernames, access tokens, API keys, passwords, Kaggle
  credentials, private hardware details, or personal filesystem paths.
- The tracked notebook uses YOUR_GITHUB_USER.
- START_FREE_TRAINING.cmd derives the fork URL from Git and writes the
  configured notebook only to .local-kaggle/, which must remain ignored.
- Store training artifacts outside Git. Never commit checkpoints, ONNX exports,
  videos, logs, or generated credential files.
- START_KAGGLE_BATCH.cmd writes user-specific Kaggle API job files only under
  .kaggle-api/, which must remain ignored.

## Repository and reproducibility

- Work on codex/free-first-bdx-policy.
- Preserve the upstream remote as
  https://github.com/apirrone/Open_Duck_Playground.git.
- UPSTREAM_COMMIT records the exact upstream base. Do not change it unless the
  fork is intentionally rebased and the full workflow is revalidated.
- Use named MuJoCo bodies, never numeric body IDs.
- Keep seeds, configuration, timing, checkpoints, exports, logs, and evaluation
  reports in the artifact bundle.
- Pin the PyPI package playground to 0.0.5. This is the newest tagged MuJoCo
  Playground release that still exports collision.geoms_colliding, wrapper,
  and config.locomotion_params as required by this fork.
- Pin Brax to 0.13.0 and JAX/JAXlib to 0.6.2. Later JAX releases remove
  device_put_replicated while this Brax PPO implementation still uses it.
- Pin MuJoCo and MuJoCo MJX to 3.3.3 for the Playground 0.0.5 training stack.
- Commit uv.lock and run Kaggle setup with uv sync --locked so dependency
  resolution cannot drift between sessions.
- Parse compatibility probe values from labelled lines because first-time imports
  may print dependency download status to standard output.

## Main entry points

- Beginner launcher: START_FREE_TRAINING.cmd
- Kaggle API batch launcher: START_KAGGLE_BATCH.cmd
- Kaggle template: notebooks/free_first_bdx_walk.ipynb
- Full guide: docs/FREE_FIRST_BDX_POLICY.md
- Training stage: scripts/run_training_stage.py
- BDX reference motion review: scripts/replay_bdx_reference.py (works where
  the upstream generator's own replay_motion.py does not: no Windows
  wheels, broken under WSLg)
- Randomization audit: scripts/randomization_audit.py
- Mass-grid evaluation: scripts/evaluate_mass_grid.py
- Locomotion-incentive gate, reference and reward together:
  scripts/check_reward_locomotion_incentive.py
- Locomotion-incentive gate, reference only (subsumed by the above):
  scripts/check_reference_locomotion_incentive.py
- Per-stage achieved-speed report: scripts/report_command_tracking.py
- Push-recovery evaluation: scripts/evaluate_push_recovery.py
- Style review: scripts/make_blind_style_review.py
- Paid guard: scripts/paid_budget_guard.py
- Pipeline status: scripts/pipeline_status.py
- Session restore: scripts/restore_artifacts.py
- Reference install: scripts/install_reference_motion.py

Kaggle artifacts belong under /kaggle/working/artifacts. Always package and
download them before the session ends.

## Reference motion invariants

- joystick.py loads the imitation reference from a hardcoded repository path,
  so a fitted reference must be installed there by
  scripts/install_reference_motion.py, which refuses to install a reference
  that does not load with the expected number of command entries.
- Kaggle setup resets the repository to the fetched commit, which discards an
  installed reference. Reinstall it after any setup rerun; the style-training
  cell preflights this and fails fast if it is missing.
- The upstream generator does not honour the requested dx/dy/dtheta: it emits a
  gait roughly 4.4x faster and encodes the achieved velocity in the output
  filename. A reference keyed by the requested command makes the imitation
  reward demand a fast gait while velocity tracking demands a slow one, and
  policies trained against that contradiction stop walking while staying
  upright. generate() therefore keys motion_grid.json by the velocity measured
  from each recording's root trajectory, never by the request. Use the rekey
  command to repair a bundle fitted before this; it preserves the approved
  motions, and any stage trained against the old keys must be retrained.
- A reference must reward walking more than marching in place, and this is a
  property of the reference, not of training. The imitation reward is dominated
  by joint_pos, an unbounded quadratic on joint angles, plus a foot-contact
  match; both are satisfied by cycling the legs on the spot. When a reference's
  fore-aft leg swing is small next to its lateral sway, marching in place scores
  higher than walking, and fine-tuning reliably finds that optimum with no
  gradient path out. Two style attempts failed this way, each surviving every
  mass-grid cell at 0.002 m/s against a 0.10 m/s command. Gate every candidate
  reference with scripts/check_reference_locomotion_incentive.py before spending
  GPU on it; it scores a known-walking and a known-collapsed policy against the
  candidate and passes only when walking wins. Measured against the 220M neutral
  policy and style seed 201: the stock reference scores +1.62, and the two
  failed bdx bundles -0.78 and -1.67. walk_foot_height is the strongest lever on
  fore-aft swing, and lowering walk_com_height also helps. Do not read
  "deliberate foot lift" as a reason to reduce walk_foot_height.
- Passing that gate is necessary, not reassuring. A reference scoring near zero
  is still far weaker than the stock one, and the reward has to make up the
  difference. Treat the stock reference's margin as the target, not the
  threshold.
- walk_foot_height search (2026-09-01), screened against the 220M neutral
  policy and style seed 201 with the fixed reward config (screen fits without
  needing the human-review gate):

      walk_foot_height   total margin   imitation-only margin
      0.02 (rejected)          +0.31            -1.54
      0.04                     +0.67            -1.31
      0.05                     +0.79            -1.18
      0.06 REFUSED by generate()'s joint-range self-check, not screened

  0.04 clears the +0.25 total-margin threshold but the imitation reward alone
  is still net-negative for walking (stock is +1.62); the pass rides on the
  tracking-reward fix, not on the reference itself being good yet. This
  comparison has a ceiling worth remembering: the "known-walking" proxy is
  the existing 220M neutral policy, which only swings about 0.351 rad on its
  own (versus stock's 0.837), so it cannot demonstrate matching a reference
  demanding a bigger swing even where a policy actually trained on it would.
  If the margin stops climbing with walk_foot_height, check whether joint_pos
  is the term that stalled before concluding the lever stopped working.
- 0.06 failed generate()'s self-check (2026-09-01): the left motion's
  right_knee recorded [+0.859, +2.121] rad, 0.05 rad past the 2.071 rad
  tolerance ceiling (model range +/-1.5708 plus RANGE_TOLERANCE 0.5), and 5
  automatic walk_com_height nudges could not clear it -- correctly refused
  rather than shipped. So the ceiling on walk_foot_height alone sits between
  0.05 and 0.06. Do not retry 0.06 as-is. Lowering walk_com_height or
  feet_spacing increases required knee flexion the same way raising
  walk_foot_height does, so neither can compensate for an over-range
  walk_foot_height -- only raising walk_com_height (the opposite direction)
  could buy headroom to push foot height further, untried.
- walk_foot_height is EXHAUSTED as a lever, and cannot reach the stock
  reference's margin no matter how far it is pushed. Decomposing the imitation
  differential per term across the three screened values shows it moves
  joint_pos and nothing else:

      term          stock   foot0.02  foot0.04  foot0.05
      joint_pos     +1.501    -0.882    -0.522    -0.398
      contact       +0.162    -0.455    -0.455    -0.455
      lin_vel_xy    +0.093    -0.124    -0.124    -0.124
      ang_vel_xy    -0.131    -0.131    -0.131    -0.131
      (others)      -0.004    -0.073    -0.073    -0.072
      non-joint_pos +0.120    -0.784    -0.783    -0.782

  The non-joint_pos subtotal is frozen at -0.78 to three decimals across every
  foot height, so even a perfect joint_pos could only reach about +0.72 against
  the stock reference's +1.62. joint_pos itself improves about +0.12 per +0.01
  of foot height, which would need walk_foot_height near 0.21 to close alone --
  and 0.06 already fails the joint-range check. Raising foot height further is
  not the way to a good reference.
- The contact term is the frozen blocker, and double_support_ratio is what
  moves it. Measured 2026-09-01 at a 0.10 m/s command: the stock reference
  spends 37% of its cycle with both feet down, ours spent 15%. The imitation
  reward's contact term scores how many feet match the reference's contact
  state, and the collapsed policy never lifts a foot, so a reference that is
  rarely in double support hands it a standing match for most of the cycle.
  double_support_ratio was never in the style config -- it inherited
  placo_defaults' 0.18 -- and is now an explicit parameter at 0.5, giving 33%
  double support. single_support_duration drops to 0.18 in the same change
  purely to hold the 0.540 s period; see the cadence invariant above, and do
  not read that 0.18 as the 0.432 s failure, which had dsr 0.18 too.
- Reference lookup interpolates between recordings; it does not snap to one.
  Nearest-neighbour lookup made the reference a staircase: with the eight
  hand-picked bdx motions every command from 0.037 to 0.111 m/s was served the
  same 0.074 m/s gait, so a 0.10 m/s command was asked to imitate a gait
  translating at 74% of it while tracking_lin_vel asked for the full speed, and
  the policy settled in between. blend_for_command projects the query onto the
  segment from the nearest entry to each candidate and blends with whichever
  candidate removes the most error, so the result is always a mix of two
  reviewed motions; sampling is linear in the coefficients, so blending them is
  the same as blending the sampled trajectories. Blending with the
  second-nearest entry instead does not work on the dense stock grid, where the
  two nearest commands normally differ along an axis the query does not need.
  Axes are compared after normalising each by the span the reference covers,
  because dx runs over about +-0.15 m/s while dtheta runs over +-1.0 rad/s and
  raw Euclidean distance let a physically trivial yaw difference outweigh a
  large speed difference. A 0.10 m/s command now selects a 0.101 m/s reference
  under the stock grid and 0.100 m/s under the bdx bundle.
- The reference's own base_angular_vel z channel does not carry the commanded
  yaw rate, though its linear channel does match its dx key. The stock entry
  keyed 1.222 rad/s averages 0.041 rad/s on that channel, about 30x low. The
  imitation reward's ang_vel_z term therefore compares the robot's real gyro
  against a near-zero target and rewards not turning. This is a property of the
  recorded data, not of this fork's code; it is recorded here because it caps
  how far the imitation reward can be trusted on the yaw axis, and correcting it
  means regenerating rather than editing.
- A fine-tune inherits its restore checkpoint's gait cadence and cannot retime.
  The imitation reward's joint_pos term is an unbounded quadratic on joint
  angles, so a policy whose stride cannot phase-lock to the reference clock
  scores strictly worse than one that stops walking and sits near the reference
  mean pose; there is no gradient path from one cadence to another. A style
  reference must therefore keep the stride period of the reference its restore
  checkpoint trained on: 0.540 s. THE PERIOD IS THE INVARIANT, not any single
  parameter. It is set exactly by
  period = 2 * (10 + round(double_support_ratio * 10)) * single_support_duration / 10
  (single_support_timesteps is 10). Verified exactly on both earlier bundles:
  ssd 0.18 / dsr 0.18 gave 0.432 s, ssd 0.225 / dsr 0.18 gave 0.540 s. An
  earlier version of this file wrote that formula as 2.4 * ssd and said ssd
  "must stay at 0.225"; that was only true while double_support_ratio sat at
  placo_defaults' 0.18, and it is not the invariant. Cadence is not one of the
  five style traits. Do not treat placo_defaults.json as the baseline: its
  single_support_duration of 0.17 is not the shipped reference's cadence.
  Beware that ssd 0.18 names both a failure and the current config, and they
  differ: ssd 0.18 with dsr 0.18 gives 0.432 s, 20% short, and made style seeds
  201/202/203 march in place at 0.002 m/s against a 0.10 m/s command; ssd 0.18
  with dsr 0.5 gives exactly 0.540 s and is what configs/bdx_inspired_reference
  .json now uses. Always judge a change by the period the formula yields, never
  by ssd alone. install_reference_motion.py refuses a reference whose period
  differs from the installed one unless --allow-cadence-change says training
  restarts from scratch.
- The command grid is deliberately sparse: eight hand-picked motions, one axis
  varying at a time, so a human can review all of them. Reference lookup must
  therefore never assume a dense dx/dy/dtheta grid.
- The upstream generator disables IK joint limits, and its solver has been
  observed to land on a different local optimum for a byte-identical preset
  depending on the machine it runs on (almost certainly floating-point
  non-associativity from differing CPU/thread counts, tipping a near-boundary
  solution into an invalid branch). A configuration validated on one machine
  is therefore not proven safe on another. scripts/prepare_bdx_reference.py
  generate() checks every motion on whichever machine actually generates it,
  retrying with a small walk_com_height nudge (up to 5 times) before failing
  loudly; do not bypass this by calling the upstream generator directly.
  Re-check manually with scripts/replay_bdx_reference.py --check when
  inspecting an existing bundle.
- That check refuses broken solutions, not merely unreachable ones. The
  reference is a soft imitation target, never a trajectory the robot replays,
  so it does not have to fit inside the model's joint ranges -- and the stock
  reference that produces a walking policy does not: all 240 of its entries
  exceed the knee range, peaking near +2.01 rad against a +-1.571 limit, about
  0.44 rad over, in the natural direction. Requiring style references to stay
  inside the range held their knee swing to roughly half the stock reference's
  at the same command, which is what let marching in place outscore walking.
  The check therefore refuses a knee that bends backwards (the IK solver's
  inverted branch, seen in 2 of 240 stock entries) and any joint exceeding its
  range by more than RANGE_TOLERANCE, currently 0.5 rad, which clears the
  overshoot the working reference uses. Do not restore a strict range check
  without also showing the resulting reference still passes
  scripts/check_reference_locomotion_incentive.py.
- Because the same input can regenerate into different motions, the human
  review gate is bound to content, not to a path: generate() and --check both
  print a bundle fingerprint, and approve --expect-fingerprint refuses a bundle
  that is not the one that was reviewed. Keep generation and approval in
  separate notebook cells so approving can never regenerate.

## Reward invariants

- The reward configuration must price walking above marching in place, exactly
  as a reference must, and it is measurable offline in about a minute with
  scripts/check_reward_locomotion_incentive.py. It scores every term in
  Joystick._get_reward for a known-walking and a known-collapsed policy, and
  prints the imitation breakdown beside the total, so one run answers both "is
  this reference safe to train on" and "is this reward safe to train with".
  Prefer it over check_reference_locomotion_incentive.py, which reports only the
  imitation half. Run it before changing a weight, and before spending GPU:

      uv run python scripts/check_reward_locomotion_incentive.py           --reference REFERENCE.pkl --bundle kaggle-outputs/SOME_BUNDLE           --cache artifacts/reward_incentive_rollouts.json

  --bundle takes both policies from an artifact bundle (furthest-trained neutral
  stage, first style seed) so neither checkpoint path has to be pasted. Each
  policy is rolled out once and cached, and everything that is not a rollout --
  every weight, both tracking sigmas, and the imitation betas -- is scored from
  that cache for free. Adding a new sweepable knob means caching its raw input
  in rollout() and recomputing it in score(), never re-rolling.
- tracking_ang_vel used to share tracking_sigma with tracking_lin_vel. Walking
  swings the torso in yaw at about 0.13 rad/s RMS about a correct mean, and
  under sigma^2 = 0.01 that reads as a 1.3-sigma error, so the term paid a
  policy standing still about 2.4 reward per step more than a walking one. That
  was the largest anti-locomotion term in the whole reward -- larger than the
  imitation differential under either reference. ang_tracking_sigma is now 0.25,
  the value the mujoco_playground G1 and T1 joystick tasks use, which prices a
  commanded turn without taxing the gait that delivers it. Do not fold it back
  into tracking_sigma.
- tracking_lin_vel is 6.0, equal to tracking_ang_vel, because linear and angular
  velocity tracking are equally the task. At the inherited 2.5 the reward paid
  more than twice as much for holding a heading as for going anywhere.
- Under these weights walking beats marching by +3.6 at a 0.10 m/s command and
  +4.5 at 0.15 against the stock reference; the inherited weights scored +0.09
  and +0.10 there. The configuration that trains a walking policy was within
  noise of preferring stillness, which is why a slightly worse reference tipped
  it over twice.
- alive cannot change this ordering. Both policies survive, so both collect it
  in full: changing it moves the margin's share of the per-step total but not
  its sign. Do not spend a run on it expecting the collapse to lift.
- Do not wire reward_feet_air_time, reward_feet_phase, cost_feet_clearance,
  cost_feet_height or cost_feet_slip into _get_reward on the strength of what
  the comparator projects do. Measured against our collapsed policy they do not
  discriminate, because it never leaves the ground at all -- zero flight phases
  in twelve seconds -- so every term gated on first contact is identically zero
  for it, earning nothing and paying nothing. feet_air_time at weight 3.0 moves
  the margin by +0.009. All five together at comparator weights make it worse,
  because feet_clearance, feet_height and feet_slip each charge the walking
  policy for motion the collapsed one does not produce.
- Rescaling the imitation reward's beta values does not help either. The
  premise that exp(-8*err^2) spans only 0.835-1.0 does not survive measurement:
  the term sees the instantaneous velocity error, which oscillates, so
  lin_vel_xy actually runs about 0.52 walking against 0.64 marching. Sweeping
  the linear beta from 8 to 200 moves the stock margin by at most +0.05, and
  makes the bdx bundle monotonically worse, +0.31 to -0.03, because sharpening
  the exponential amplifies terms that already favour standing still.
- The neutral policy has a low-command dead zone that the reward correction is
  meant to close, and it is the cheapest acceptance check available. Under the
  old reward, measured forward speed was:

      stage                     cmd 0.05   cmd 0.10   cmd 0.15
      20M nominal                 0.0013     0.0014     0.0021
      80M moderate                0.0019     0.0167     0.0929
      300M full                   0.0013     0.0645     0.0925

  It does not walk at all until somewhere past 20M, and never walks at 0.05 m/s
  at any stage. Compare a retrained line against this table rather than against
  the command. The 20M row is too close to zero to read; 80M is the first
  informative checkpoint, and cmd 0.05 is the sharpest signal because the old
  reward never got the policy moving there.
- Tested (2026-09-01): retraining the full 300M neutral line under the fixed
  reward and interpolated reference. The cmd 0.05 dead zone did NOT close, and
  cmd 0.10 at 300M did not meaningfully improve:

      stage           cmd 0.05          cmd 0.10          cmd 0.15
      20M nominal    0.0013 -> 0.0015  0.0014 -> 0.0014  0.0021 -> 0.0017
      80M moderate   0.0019 -> 0.0018  0.0167 -> 0.0535  0.0929 -> 0.1103
      300M full      0.0013 -> 0.0008  0.0645 -> 0.0652  0.0925 -> 0.1103

  (old -> new, uv run python scripts/report_command_tracking.py). At 300M, cmd
  0.10 moved +1% and cmd 0.05 got slightly worse -- both changes are noise, not
  signal. The real effect is entirely at 80M: cmd 0.10 there is 3.2x faster
  under the new reward (0.0167 -> 0.0535), i.e. the new reward reaches a usable
  gait in far fewer steps, but that lead almost entirely evaporates by 300M
  because the OLD reward closes most of the gap over the same later steps
  (0.0167 -> 0.0645 old, vs 0.0535 -> 0.0652 new). Both configurations converge
  to nearly the same terminal cmd-0.10 speed by 300M. Read this as: the reward
  fix is confirmed to prevent the marching-in-place collapse and to speed up
  early convergence, but it is NOT confirmed to raise the neutral policy's
  asymptotic cruising speed, and the cmd 0.05 dead zone specifically remains
  unexplained and unfixed. Do not claim the reward fix solved the velocity
  deficit; it did not, only the collapse. cmd 0.15 improved a real, modest
  +19% at every stage, which is the one number here that unambiguously moved.
  Something else (a genuine cadence/amplitude ceiling from the fixed 0.540 s
  stride period, or full-randomization robustness trading away speed) likely
  caps cmd-0.10 cruising speed independent of reward shape; not yet
  investigated. This does not block moving on to the reference/style stages --
  the neutral checkpoint is not regressed, only unimproved at cmd 0.10/0.05 --
  but do not re-cite the earlier prediction of a clear win there as if it held.
- A stage is only interchangeable with an earlier one if it trained against the
  same objective. stage_result.json therefore records a reward_config
  fingerprint of the whole reward_config block, and run_training_stage.py
  refuses to reuse a stage whose fingerprint differs. Every stage completed
  before the weights above will retrain.

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
- File persistence may restore .venv without a usable executable or symlink.
  Setup must recreate only .venv with uv venv --clear in that condition.
- Every notebook code cell starts with a "# ==== CELL N: name ====" header,
  numbered in run order. Keep the numbering contiguous and update it when
  cells are added, removed, or reordered, and refer to cells by that number in
  guides and in conversation so instructions are unambiguous.
- The header is followed by a second comment line estimating that cell's
  run time from scratch: "# Measured this session (T4 x2): ..." once a real
  Kaggle run has timed it, "# Estimated time: ..." otherwise, with the basis
  stated (scaled from which other measured stage, or genuinely unmeasured).
  Update the measured lines from the actual elapsed_seconds in
  stage_result.json once a stage completes, rather than leaving a stale
  estimate standing next to a real number. This applies to any new Kaggle
  cell offered in conversation too, not only the tracked notebook.
- Do not use Save & Run All while developing because the notebook contains
  benchmark and human-review gates.
- A notebook cell must never depend on a Python variable defined by an earlier
  cell for pipeline state. A Kaggle restart clears the kernel while
  /kaggle/working survives, so later cells read completed stages from disk via
  load_stage(); only then does resuming actually work.
- Reuse a complete stage only when its full configuration matches and both its
  checkpoint and ONNX export still exist. Never treat a partial stage or a
  missing artifact as complete.
- Kaggle API batch runs are non-interactive clean workers. They are useful for
  smoke and benchmark automation, but they do not replace interactive file
  persistence or human reference-motion review gates.

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
- For reward or reference-lookup changes, run
  scripts/check_reward_locomotion_incentive.py against both the stock reference
  and the installed one, and report the margins. A change that does not improve
  them is not a fix, whatever else recommends it.
- Run the 10,000-model audit before full training.
- Confirm the configured .local-kaggle notebook is ignored.
- Scan tracked files for credential-like strings and personal identifiers.
- Keep the working tree clean and report the published commit.

## Failure recovery

On NaNs or reward collapse, resume the last good checkpoint with COM offsets
halved. If failure repeats, return to moderate mass ranges before changing
rewards.

A successful stage retry must remove its stale stage_failure.json marker.
