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


def verify(path: Path, expected_entries: int | None) -> int:
    """Load the reference the same way training does and report entry count."""
    from playground.common.poly_reference_motion import PolyReferenceMotion

    prm = PolyReferenceMotion(str(path))
    entries = int(prm.data_array.shape[0])
    if expected_entries is not None and entries != expected_entries:
        raise SystemExit(
            f"Reference at {path} has {entries} command entries, expected {expected_entries}"
        )
    if prm.nb_steps_in_period is None or prm.nb_steps_in_period <= 0:
        raise SystemExit(f"Reference at {path} has invalid nb_steps_in_period")
    return entries


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
    args = parser.parse_args()

    if args.source:
        source = args.source.resolve()
        if not source.is_file():
            raise SystemExit(f"Fitted reference not found: {source}")
        # Verify before installing so a broken file never lands in the repo.
        entries = verify(source, args.expect_entries)
        TRAINING_REFERENCE.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, TRAINING_REFERENCE)
        print(f"Installed {source} -> {TRAINING_REFERENCE} ({entries} command entries)")

    if not TRAINING_REFERENCE.is_file():
        raise SystemExit(f"No reference installed at {TRAINING_REFERENCE}")

    entries = verify(TRAINING_REFERENCE, args.expect_entries)
    print(f"Verified training reference: {TRAINING_REFERENCE} ({entries} command entries)")


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    main()
