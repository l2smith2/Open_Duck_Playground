import json
import sys

from scripts import run_training_stage


def test_successful_retry_removes_stale_failure_marker(tmp_path, monkeypatch):
    output = tmp_path / "stage"
    output.mkdir()
    failure = output / "stage_failure.json"
    failure.write_text('{"status": "failed"}\n', encoding="utf-8")
    checkpoint = output / "checkpoint_1"
    checkpoint.mkdir()
    onnx = output / "policy.onnx"
    onnx.write_bytes(b"test")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_training_stage.py",
            "--name",
            "retry",
            "--output-dir",
            str(output),
            "--steps",
            "1",
            "--randomization-stage",
            "nominal",
            "--seed",
            "7",
        ],
    )
    monkeypatch.setattr(
        run_training_stage.subprocess, "run", lambda *args, **kwargs: None
    )

    run_training_stage.main()

    assert not failure.exists()
    result = json.loads((output / "stage_result.json").read_text(encoding="utf-8"))
    assert result["status"] == "complete"
    assert result["checkpoint"] == str(checkpoint.resolve())
    assert result["onnx"] == str(onnx.resolve())
    assert result["final_training_step"] == 1


def test_completed_stage_is_reused_without_training(tmp_path, monkeypatch):
    output = tmp_path / "stage"
    output.mkdir()
    checkpoint = output / "checkpoint_2293760"
    checkpoint.mkdir()
    onnx = output / "policy.onnx"
    onnx.write_bytes(b"test")
    result = {
        "name": "smoke",
        "status": "complete",
        "steps_added": 1_000_000,
        "elapsed_seconds": 10.0,
        "randomization_stage": "nominal",
        "seed": 100,
        "restore": None,
        "checkpoint": str(checkpoint.resolve()),
        "onnx": str(onnx.resolve()),
        "imitation_reward_weight_scale": 1.0,
        "com_offset_scale": 1.0,
        "reward_config": run_training_stage.reward_config_fingerprint(),
    }
    (output / "stage_result.json").write_text(
        json.dumps(result), encoding="utf-8"
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_training_stage.py",
            "--name",
            "smoke",
            "--output-dir",
            str(output),
            "--steps",
            "1000000",
            "--randomization-stage",
            "nominal",
            "--seed",
            "100",
        ],
    )

    def unexpected_training(*args, **kwargs):
        raise AssertionError("completed stage should not train again")

    monkeypatch.setattr(run_training_stage.subprocess, "run", unexpected_training)

    run_training_stage.main()


def test_stage_trained_under_a_different_reward_is_not_reused(tmp_path, monkeypatch):
    # A checkpoint is only interchangeable with an earlier one if it was
    # optimised against the same objective. Reusing across a reward change would
    # quietly ship a policy tuned for weights nobody is training with any more.
    output = tmp_path / "stage"
    output.mkdir()
    checkpoint = output / "checkpoint_2293760"
    checkpoint.mkdir()
    onnx = output / "policy.onnx"
    onnx.write_bytes(b"test")
    result = {
        "name": "smoke",
        "status": "complete",
        "steps_added": 1_000_000,
        "elapsed_seconds": 10.0,
        "randomization_stage": "nominal",
        "seed": 100,
        "restore": None,
        "checkpoint": str(checkpoint.resolve()),
        "onnx": str(onnx.resolve()),
        "imitation_reward_weight_scale": 1.0,
        "com_offset_scale": 1.0,
        "reward_config": "staleconfig01",
    }
    (output / "stage_result.json").write_text(json.dumps(result), encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_training_stage.py",
            "--name",
            "smoke",
            "--output-dir",
            str(output),
            "--steps",
            "1000000",
            "--randomization-stage",
            "nominal",
            "--seed",
            "100",
        ],
    )
    trained = []
    monkeypatch.setattr(
        run_training_stage.subprocess, "run", lambda *a, **k: trained.append(a)
    )

    run_training_stage.main()

    assert trained, "a stage from a different reward config must be retrained"
    rewritten = json.loads((output / "stage_result.json").read_text(encoding="utf-8"))
    assert rewritten["reward_config"] == run_training_stage.reward_config_fingerprint()
