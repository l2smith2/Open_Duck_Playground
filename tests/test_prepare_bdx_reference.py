import json

import pytest

from scripts import prepare_bdx_reference as pbr


def _make_generator_root(tmp_path):
    generator_root = tmp_path / "generator"
    gen_pkg = generator_root / "open_duck_reference_motion_generator"
    gen_pkg.mkdir(parents=True)
    (gen_pkg / "gait_generator.py").write_text("# stub, subprocess.run is mocked in tests\n")
    robot_dir = gen_pkg / "robots" / "open_duck_mini_v2"
    robot_dir.mkdir(parents=True)
    (robot_dir / "placo_defaults.json").write_text(json.dumps({
        "dx": 0.0, "dy": 0.0, "dtheta": 0.0, "walk_com_height": 0.2,
        "walk_foot_height": 0.04, "walk_trunk_pitch": -3.0,
        "single_support_duration": 0.17, "feet_spacing": 0.16,
    }))
    return generator_root


class FakeChecker:
    """Returns a violation for the first `bad_attempts` checks of each motion, then clean."""

    def __init__(self, bad_attempts_per_motion):
        self.bad_attempts = dict(bad_attempts_per_motion)
        self.calls = []

    def check_file(self, recording):
        name = recording.stem.removeprefix("bdx_inspired_").rsplit("_", 3)[0]
        self.calls.append(name)
        remaining = self.bad_attempts.get(name, 0)
        if remaining > 0:
            self.bad_attempts[name] = remaining - 1
            return [f"  {recording.name}: fake_joint out of range"]
        return []


# The generator does not honour the requested velocity; it produces a faster
# gait and names the file after what it achieved. Fakes reproduce that so the
# tests exercise the real relationship between request and result.
GENERATOR_SPEEDUP = 4.4


def _write_fake_motion(path, dx, dy, fps=50, count=100):
    frames = []
    for index in range(count):
        seconds = index / fps
        frames.append([dx * seconds, dy * seconds, 0.2, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    path.write_text(json.dumps({
        "FPS": fps,
        "Frame_offset": [{"root_pos": 0, "root_quat": 3, "joints_pos": 7}],
        "Joints": ["a", "b"],
        "Frames": frames,
    }))


def _patch_generation(monkeypatch, artifact_dir, checker):
    """Make generate_one's subprocess call a no-op that writes the expected output file."""

    def fake_run(command, cwd, check):
        # command: [uv, run, script, --duck, ..., --output_dir, DIR, --length, ..., --preset, PRESET, --name, NAME]
        output_dir = artifact_dir / "recordings"
        name = command[command.index("--name") + 1]
        preset_path = command[command.index("--preset") + 1]
        preset = json.loads(open(preset_path).read())
        achieved_dx = preset["dx"] * GENERATOR_SPEEDUP
        achieved_dy = preset["dy"] * GENERATOR_SPEEDUP
        out = output_dir / f"{name}_{preset['dx']}_{preset['dy']}_{preset['dtheta']}.json"
        _write_fake_motion(out, achieved_dx, achieved_dy)
        return None

    monkeypatch.setattr(pbr.subprocess, "run", fake_run)
    monkeypatch.setattr(pbr, "JointRangeChecker", lambda: checker)


def test_generate_retries_and_succeeds_before_exhausting_attempts(tmp_path, monkeypatch):
    generator_root = _make_generator_root(tmp_path)
    artifact_dir = tmp_path / "artifact"
    monkeypatch.setattr(pbr, "MOTION_GRID", (("stand", 0.0, 0.0, 0.0),))
    checker = FakeChecker({"stand": 2})  # bad twice, then clean on the 3rd generation
    _patch_generation(monkeypatch, artifact_dir, checker)

    pbr.generate(generator_root, artifact_dir)

    assert checker.calls == ["stand", "stand", "stand"]
    motion_grid = json.loads((artifact_dir / "motion_grid.json").read_text())
    assert len(motion_grid) == 1


def test_generate_raises_after_exhausting_retries(tmp_path, monkeypatch):
    generator_root = _make_generator_root(tmp_path)
    artifact_dir = tmp_path / "artifact"
    monkeypatch.setattr(pbr, "MOTION_GRID", (("stand", 0.0, 0.0, 0.0),))
    checker = FakeChecker({"stand": 999})  # never clean
    _patch_generation(monkeypatch, artifact_dir, checker)

    with pytest.raises(RuntimeError, match="persisted after 5 retries"):
        pbr.generate(generator_root, artifact_dir)

    # initial attempt + 5 retries = 6 generations, every one checked
    assert len(checker.calls) == 6


def test_generate_succeeds_on_final_retry_without_extra_unchecked_generation(tmp_path, monkeypatch):
    # Regression test: an earlier version of the retry loop generated a 6th
    # candidate on the final retry but never checked it before raising.
    generator_root = _make_generator_root(tmp_path)
    artifact_dir = tmp_path / "artifact"
    monkeypatch.setattr(pbr, "MOTION_GRID", (("stand", 0.0, 0.0, 0.0),))
    checker = FakeChecker({"stand": 5})  # bad for attempts 1-5, clean on the 6th generation
    _patch_generation(monkeypatch, artifact_dir, checker)

    pbr.generate(generator_root, artifact_dir)  # must not raise

    assert len(checker.calls) == 6


def _make_bundle(tmp_path, count=8):
    recordings = tmp_path / "recordings"
    recordings.mkdir(parents=True)
    for index in range(count):
        (recordings / f"motion_{index}.json").write_text(f'{{"n": {index}}}')
    return tmp_path


def test_approve_accepts_the_fingerprint_that_was_reviewed(tmp_path):
    artifact_dir = _make_bundle(tmp_path)
    fingerprint = pbr.bundle_fingerprint(artifact_dir / "recordings")

    pbr.approve(artifact_dir, "Replayed all eight motions", fingerprint)

    marker = json.loads((artifact_dir / "reference_review_approved.json").read_text())
    assert marker["approved"] is True
    assert marker["fingerprint"] == fingerprint


def test_approve_refuses_a_bundle_that_was_not_the_one_reviewed(tmp_path):
    # Regenerating produces different motion data; approval must not carry over.
    artifact_dir = _make_bundle(tmp_path)
    stale = pbr.bundle_fingerprint(artifact_dir / "recordings")
    (artifact_dir / "recordings" / "motion_0.json").write_text('{"n": 999}')
    assert pbr.bundle_fingerprint(artifact_dir / "recordings") != stale

    with pytest.raises(RuntimeError, match="not the ones you watched"):
        pbr.approve(artifact_dir, "Replayed all eight motions", stale)

    assert not (artifact_dir / "reference_review_approved.json").exists()


def test_generate_keys_motions_by_measured_velocity_not_requested(tmp_path, monkeypatch):
    # Regression test: keying by the requested command made the imitation
    # reward demand a gait 4.4x faster than the velocity-tracking reward, and
    # policies trained against that contradiction stopped walking entirely.
    generator_root = _make_generator_root(tmp_path)
    artifact_dir = tmp_path / "artifact"
    monkeypatch.setattr(pbr, "MOTION_GRID", (("forward", 0.04, 0.0, 0.0),))
    _patch_generation(monkeypatch, artifact_dir, FakeChecker({}))

    pbr.generate(generator_root, artifact_dir)

    grid = json.loads((artifact_dir / "motion_grid.json").read_text())
    entry = next(iter(grid.values()))
    assert entry["dx"] == pytest.approx(0.04 * GENERATOR_SPEEDUP, abs=1e-3)
    assert entry["dx"] != pytest.approx(0.04, abs=1e-3)


def test_rekey_refiles_an_existing_bundle_without_touching_the_motions(tmp_path, monkeypatch):
    generator_root = _make_generator_root(tmp_path)
    artifact_dir = tmp_path / "artifact"
    monkeypatch.setattr(pbr, "MOTION_GRID", (("forward", 0.04, 0.0, 0.0),))
    _patch_generation(monkeypatch, artifact_dir, FakeChecker({}))
    pbr.generate(generator_root, artifact_dir)

    recordings = sorted((artifact_dir / "recordings").glob("*.json"))
    before = pbr.bundle_fingerprint(artifact_dir / "recordings")

    # Simulate a bundle fitted under the old, requested-command keys.
    stale = {recordings[0].name: {"dx": 0.04, "dy": 0.0, "dtheta": 0.0}}
    (artifact_dir / "motion_grid.json").write_text(json.dumps(stale))
    import pickle
    with open(artifact_dir / "polynomial_coefficients.pkl", "wb") as handle:
        pickle.dump({"0.04_0.0_0.0": "coefficients"}, handle)

    pbr.rekey(artifact_dir, None)

    with open(artifact_dir / "polynomial_coefficients.pkl", "rb") as handle:
        rekeyed = pickle.load(handle)
    (key,) = rekeyed
    assert rekeyed[key] == "coefficients"
    assert float(key.split("_")[0]) == pytest.approx(0.04 * GENERATOR_SPEEDUP, abs=1e-3)
    # The approved motion data itself must be untouched.
    assert pbr.bundle_fingerprint(artifact_dir / "recordings") == before
