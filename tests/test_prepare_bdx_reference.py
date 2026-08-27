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


def _patch_generation(monkeypatch, artifact_dir, checker):
    """Make generate_one's subprocess call a no-op that writes the expected output file."""

    def fake_run(command, cwd, check):
        # command: [uv, run, script, --duck, ..., --output_dir, DIR, --length, ..., --preset, PRESET, --name, NAME]
        output_dir = artifact_dir / "recordings"
        name = command[command.index("--name") + 1]
        preset_path = command[command.index("--preset") + 1]
        preset = json.loads(open(preset_path).read())
        out = output_dir / f"{name}_{preset['dx']}_{preset['dy']}_{preset['dtheta']}.json"
        out.write_text("{}")
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
