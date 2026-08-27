"""Restore a previous session's artifact bundle from an attached Kaggle dataset.

Kaggle's interactive file persistence is best effort: a wedged or expired
session can leave /kaggle/working empty, and a freshly imported notebook always
starts empty. Uploading the last artifact ZIP as a private Kaggle Dataset and
attaching it as an input makes those artifacts recoverable without retraining.

Kaggle expands an uploaded ZIP into the dataset, so an attached input usually
holds the extracted directory tree rather than the ZIP itself -- but a ZIP can
also arrive verbatim. Both are handled here. The stage directories may also sit
at the top level (bundles produced by save_bundle are rooted at the artifacts
directory) or one level deeper (hand-made bundles), so their location is
discovered rather than assumed.

Existing artifacts are never overwritten unless --force is given, so restoring
can not silently clobber work that is newer than the backup.
"""

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path


def artifacts_root_within(tree: Path) -> Path | None:
    """Locate the directory holding the stage directories, at any depth."""
    stage_dirs = [p.parent for p in tree.rglob("stage_result.json")]
    if not stage_dirs:
        return None
    roots = {d.parent for d in stage_dirs}
    if len(roots) == 1:
        return roots.pop()
    # Stages at differing depths: prefer the shallowest common location.
    return min(roots, key=lambda p: len(p.relative_to(tree).parts))


def install_from(root: Path, artifacts: Path, force: bool) -> tuple[list[str], list[str]]:
    """Copy everything in root into artifacts, skipping what already exists."""
    restored: list[str] = []
    skipped: list[str] = []
    artifacts.mkdir(parents=True, exist_ok=True)
    for item in sorted(root.iterdir()):
        target = artifacts / item.name
        if target.exists():
            if not force:
                skipped.append(item.name)
                continue
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        # Copy rather than move: /kaggle/input is read-only.
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
        restored.append(item.name)
    return restored, skipped


def restore_from_zip(zip_path: Path, artifacts: Path, force: bool):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_path)
        root = artifacts_root_within(tmp_path)
        if root is None:
            return None
        return install_from(root, artifacts, force)


def find_sources(input_root: Path) -> list[tuple[str, Path]]:
    """Attached inputs that look like an artifact bundle, ZIP or extracted."""
    sources: list[tuple[str, Path]] = []
    if not input_root.is_dir():
        return sources
    for dataset in sorted(p for p in input_root.iterdir() if p.is_dir()):
        for zip_path in sorted(dataset.rglob("*.zip")):
            sources.append(("zip", zip_path))
        if artifacts_root_within(dataset) is not None:
            sources.append(("dir", dataset))
    return sources


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("/kaggle/working/artifacts"))
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--zip", type=Path, help="Restore this ZIP instead of scanning inputs")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite artifacts that already exist (default keeps what is already there)",
    )
    args = parser.parse_args()

    sources = [("zip", args.zip)] if args.zip else find_sources(args.input_root)
    if not sources:
        print(f"No artifact bundle found under {args.input_root}.")
        print("Attach your backup dataset with Add Input, or skip this if the artifacts are already present.")
    for kind, path in sources:
        if kind == "zip":
            outcome = restore_from_zip(path, args.artifacts, args.force)
        else:
            root = artifacts_root_within(path)
            outcome = install_from(root, args.artifacts, args.force) if root else None
        if outcome is None:
            print(f"{path.name}: no stage directories inside, ignoring")
            continue
        restored, skipped = outcome
        print(f"{path.name} ({kind}): restored {len(restored)}, kept existing {len(skipped)}")
        for name in restored:
            print(f"  + {name}")
        for name in skipped:
            print(f"  = {name} (already present, not overwritten)")

    present = sorted(p.name for p in args.artifacts.iterdir()) if args.artifacts.is_dir() else []
    print(f"\nArtifacts now present ({len(present)}): {present}")


if __name__ == "__main__":
    main()
