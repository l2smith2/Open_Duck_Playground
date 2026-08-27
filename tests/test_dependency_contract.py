import json
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_STACK = {
    "brax": "0.13.0",
    "jax": "0.6.2",
    "jax-cuda12-plugin": "0.6.2",
    "jaxlib": "0.6.2",
    "mujoco": "3.3.3",
    "mujoco-mjx": "3.3.3",
    "playground": "0.0.5",
}


def _base_dependency(spec: str) -> str:
    """Strip a PEP 508 environment marker (e.g. "; sys_platform == 'linux'")."""
    return spec.split(";")[0].strip()


def test_training_dependencies_and_lockfile_match():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = {_base_dependency(d) for d in project["project"]["dependencies"]}
    assert "brax==0.13.0" in dependencies
    # jax[cuda12] is Linux/Kaggle-only (no Windows wheels for the cuda12
    # extra); plain jax covers local Windows development. Both must pin the
    # same version as the training stack.
    assert "jax[cuda12]==0.6.2" in dependencies
    assert "jax==0.6.2" in dependencies
    assert "jaxlib==0.6.2" in dependencies
    assert "mujoco==3.3.3" in dependencies
    assert "mujoco-mjx==3.3.3" in dependencies
    assert "playground==0.0.5" in dependencies

    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    locked = {package["name"]: package["version"] for package in lock["package"]}
    for name, version in EXPECTED_STACK.items():
        assert locked[name] == version


def test_kaggle_setup_uses_and_verifies_locked_stack():
    notebook = json.loads(
        (ROOT / "notebooks" / "free_first_bdx_walk.ipynb").read_text(
            encoding="utf-8"
        )
    )
    setup = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "EXPECTED_UPSTREAM" in "".join(cell.get("source", []))
    )
    assert 'subprocess.run(["uv", "sync", "--locked"]' in setup
