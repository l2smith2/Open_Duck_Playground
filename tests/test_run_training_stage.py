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
