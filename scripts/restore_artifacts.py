"""Restore a previous session's artifact bundle from an attached Kaggle dataset.

Kaggle's interactive file persistence is best effort: a wedged or expired
session can leave /kaggle/working empty, and a freshly imported notebook always
starts empty. Uploading the last artifact ZIP as a private Kaggle Dataset and
attaching it as an input makes those artifacts recoverable without retraining.

This scans the attached inputs for an artifact ZIP, works out where the stage
directories live inside it (bundles produced by save_bundle are rooted at the
artifacts directory itself; hand-made ones are often nested one level deeper),
and moves anything missing into the artifacts directory. Existing stages are
left alone unless --force is given, so restoring can never silently overwrite
work that is newer than the backup.
"""

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path


def find_candidate_zips(input_root: Path) -> list[Path]:
    if not input_root.is_dir():
        return []
    return sorted(p for p in input_root.glob("*/*.zip") if p.is_file())


def artifacts_root_within(extracted: Path) -> Path | None:
    """Locate the directory that holds the stage directories."""
    stage_dirs = [p.parent for p in extracted.rglob("stage_result.json")]
    if not stage_dirs:
        return None
    roots = {d.parent for d in stage_dirs}
    if len(roots) == 1:
        return roots.pop()
    # Stages somehow live at different depths; fall back to the shallowest.
    return min(roots, key=lambda p: len(p.relative_to(extracted).parts))


def restore(zip_path: Path, artifacts: Path, force: bool) -> tuple[list[str], list[str]]:
    restored: list[str] = []
    skipped: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_path)
        root = artifacts_root_within(tmp_path)
        if root is None:
            return restored, skipped
        artifacts.mkdir(parents=True, exist_ok=True)
        for item in sorted(root.iterdir()):
            target = artifacts / item.name
            if target.exists():
                if not force:
                    skipped.append(item.name)
                    continue
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(item), str(target))
            restored.append(item.name)
    return restored, skipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("/kaggle/working/artifacts"))
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--zip", type=Path, help="Restore this ZIP instead of scanning inputs")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite artifacts that already exist (default is to keep what is already there)",
    )
    args = parser.parse_args()

    zips = [args.zip] if args.zip else find_candidate_zips(args.input_root)
    if not zips:
        print(f"No artifact ZIP found under {args.input_root}.")
        print("Attach your backup dataset via Add Input, or skip this if /kaggle/working already has the artifacts.")
        return

    total_restored: list[str] = []
    for zip_path in zips:
        restored, skipped = restore(zip_path, args.artifacts, args.force)
        if not restored and not skipped:
            print(f"{zip_path.name}: no stage directories inside, ignoring")
            continue
        print(f"{zip_path.name}: restored {len(restored)}, kept existing {len(skipped)}")
        for name in restored:
            print(f"  + {name}")
        for name in skipped:
            print(f"  = {name} (already present, not overwritten)")
        total_restored.extend(restored)

    present = sorted(p.name for p in args.artifacts.iterdir()) if args.artifacts.is_dir() else []
    print(f"\nArtifacts now present ({len(present)}): {present}")


if __name__ == "__main__":
    main()
