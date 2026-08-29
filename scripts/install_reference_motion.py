"""Install a fitted reference motion into the training data path, and verify it.

joystick.py loads the imitation reference from a hardcoded path inside the
repository (playground/open_duck_mini_v2/data/polynomial_coefficients.pkl).
That file is tracked by Git, so a fitted style reference copied over it can be
silently reverted -- or silently kept stale -- by a later branch update,
because Git leaves a locally-modified file alone when its content is identical
between the old and new commits.

This script makes installation explicit and re-runnable: copy the fitted
reference from the artifact bundle into place, then prove it actually loads and
exposes the expected number of command entries before any training starts.
"""

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_REFERENCE = (
    REPO_ROOT / "playground" / "open_duck_mini_v2" / "data" / "polynomial_coefficients.pkl"
)


def verify(path: Path, expected_entries: int | None) -> tuple[int, float]:
    """Load the reference the same way training does; report entries and stride period."""
    from playground.common.poly_reference_motion import PolyReferenceMotion

    prm = PolyReferenceMotion(str(path))
    entries = int(prm.data_array.shape[0])
    if expected_entries is not None and entries != expected_entries:
        raise SystemExit(
            f"Reference at {path} has {entries} command entries, expected {expected_entries}"
        )
    if prm.nb_steps_in_period is None or prm.nb_steps_in_period <= 0:
        raise SystemExit(f"Reference at {path} has invalid nb_steps_in_period")
    return entries, float(prm.period)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        help="Fitted polynomial_coefficients.pkl to install (omit to only verify what is installed)",
    )
    parser.add_argument(
        "--expect-entries",
        type=int,
        default=None,
        help="Fail unless the reference exposes exactly this many command entries",
    )
    parser.add_argument(
        "--allow-cadence-change",
        action="store_true",
        help=(
            "install even though the new reference has a different stride period than the "
            "one already installed; only valid when training restarts from scratch"
        ),
    )
    args = parser.parse_args()

    if args.source:
        source = args.source.resolve()
        if not source.is_file():
            raise SystemExit(f"Fitted reference not found: {source}")
        # Verify before installing so a broken file never lands in the repo.
        entries, period = verify(source, args.expect_entries)
        # A fine-tune inherits the restore checkpoint's gait cadence. The imitation
        # reward's joint_pos term is an unbounded quadratic on joint angles, so a
        # policy whose stride cannot phase-lock to the reference clock scores worse
        # than one that stops walking and sits near the reference mean pose. Swapping
        # in a reference with a different period therefore does not retime the gait,
        # it collapses it to marching in place -- which is exactly what happened to
        # style seeds 201/202/203 when a 0.432 s reference replaced a 0.540 s one.
        if TRAINING_REFERENCE.is_file() and not args.allow_cadence_change:
            _, installed_period = verify(TRAINING_REFERENCE, None)
            # 5 ms absorbs float noise and the generator's timestep quantisation
            # while still catching the 108 ms error that stopped the first seeds.
            if abs(installed_period - period) > 0.005:
                raise SystemExit(
                    f"Refusing to install: {source} has a stride period of {period:.3f}s, "
                    f"but the reference already installed has {installed_period:.3f}s. Any "
                    "checkpoint trained on the installed reference is locked to its cadence "
                    "and will stop walking rather than retime. Match single_support_duration "
                    f"to the installed cadence (period = 2.4 * single_support_duration, so "
                    f"{installed_period / 2.4:.4f}) and regenerate, or pass "
                    "--allow-cadence-change if you are training from scratch."
                )
        TRAINING_REFERENCE.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, TRAINING_REFERENCE)
        print(
            f"Installed {source} -> {TRAINING_REFERENCE} "
            f"({entries} command entries, {period:.3f}s stride period)"
        )

    if not TRAINING_REFERENCE.is_file():
        raise SystemExit(f"No reference installed at {TRAINING_REFERENCE}")

    entries, period = verify(TRAINING_REFERENCE, args.expect_entries)
    print(
        f"Verified training reference: {TRAINING_REFERENCE} "
        f"({entries} command entries, {period:.3f}s stride period)"
    )


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    main()
