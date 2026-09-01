from scripts.report_command_tracking import discover_stages, latest_onnx


def test_latest_onnx_picks_the_highest_step_count(tmp_path):
    for step in (1_474_560, 163_840, 2_949_120):
        (tmp_path / f"2026_01_01_000000_{step}.onnx").write_bytes(b"x")
    assert latest_onnx(tmp_path).name.endswith("_2949120.onnx")


def test_latest_onnx_is_none_for_an_empty_stage(tmp_path):
    assert latest_onnx(tmp_path) is None


def test_discover_stages_follows_pipeline_status_order_and_skips_gaps(tmp_path):
    # Mirrors a real bundle: the neutral line finished, style seeds have not
    # started yet. discover_stages must not report a hole as a stage.
    for name in ("00_smoke_1m", "01_neutral_nominal_20m", "03_neutral_full_220m"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "2026_01_01_000000_1.onnx").write_bytes(b"x")
    # 02_neutral_moderate_60m deliberately absent, as if it had not run yet.
    found = discover_stages(tmp_path)
    names = [name for name, _onnx in found]
    assert names == ["00_smoke_1m", "01_neutral_nominal_20m", "03_neutral_full_220m"]


def test_discover_stages_on_an_empty_artifacts_dir(tmp_path):
    assert discover_stages(tmp_path) == []
