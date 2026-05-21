"""Rewrite every task.toml's docker_image to point at our public registry.

After ``prepare_harbor_dataset.py`` extracts Nemotron-Terminal-Synthetic-Tasks,
each ``task.toml`` ships:

    [environment]
    docker_image = "gitlab-master.nvidia.com:5005/renjiep/datagen-flash-images/hb__datagen-flash-<skill>:latest"

— a private NVIDIA gitlab registry that we (and e2b) can't pull. This script
re-points each one at our public ghcr.io build (see scripts/build_skill_images.sh).

The skill comes from the parent directory name (e.g. ``data_science/data_science_task_0914/task.toml``).

Usage:
    uv run -m examples.train_integrations.harbor_pgc.scripts.rewrite_task_dockerimage \\
        examples/train_integrations/harbor_pgc/data/Nemotron-Terminal-Synthetic-Tasks \\
        --owner rucnyz --tag 1.0
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# task.toml line we replace. Naive regex (no toml parser) keeps formatting +
# comments intact and survives the mild variations across shards.
_DOCKER_IMAGE_RE = re.compile(
    r'^(\s*docker_image\s*=\s*")[^"]*(".*)$',
    flags=re.MULTILINE,
)

_SKILL_HINTS = {
    # Subdir name → ghcr image stem. The Nemotron shards consistently nest as
    # ``<skill>/<skill>_task_<N>/`` so the immediate parent dir is the skill.
    # All 11 skills shipping across mixed / easy / medium shards.
    "data_processing",
    "data_science",
    "debugging",
    "file_operations",
    "scientific_computing",
    "security",
    # Extra in easy + medium (not in mixed/):
    "data_querying",
    "software_engineering",
    "dependency_management",
    # Extra only in medium_shard1:
    "model_training",
    "system_administration",
}


def _skill_for(task_toml: Path) -> str | None:
    """Walk up the path looking for a known skill directory name."""
    for parent in task_toml.parents:
        if parent.name in _SKILL_HINTS:
            return parent.name
    return None


def _new_image_uri(owner: str, skill: str, tag: str) -> str:
    return f"ghcr.io/{owner}/pgc-nemotron-{skill}:{tag}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "dataset_dir",
        help="Root of the prepared dataset (contains skill_based/ etc).",
    )
    p.add_argument(
        "--owner",
        default="rucnyz",
        help="ghcr.io namespace owning the pgc-nemotron-* packages.",
    )
    p.add_argument(
        "--tag",
        default="1.0",
        help="Image tag pushed by scripts/build_skill_images.sh.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change but don't write.",
    )
    args = p.parse_args()

    root = Path(args.dataset_dir).expanduser().resolve()
    if not root.is_dir():
        print(f"dataset_dir not a directory: {root}", file=sys.stderr)
        return 1

    rewritten = 0
    skipped_no_skill = 0
    skipped_no_image = 0
    for task_toml in root.rglob("task.toml"):
        skill = _skill_for(task_toml)
        if not skill:
            skipped_no_skill += 1
            continue

        original = task_toml.read_text()
        if "docker_image" not in original:
            skipped_no_image += 1
            continue

        new_uri = _new_image_uri(args.owner, skill, args.tag)
        new_text, n = _DOCKER_IMAGE_RE.subn(
            rf'\g<1>{new_uri}\g<2>',
            original,
            count=1,
        )
        if n == 0:
            # Regex missed (e.g. unusual whitespace) — skip rather than corrupt.
            skipped_no_image += 1
            continue
        if new_text == original:
            continue

        if args.dry_run:
            if rewritten < 3:
                print(f"[dry-run] would update {task_toml.relative_to(root)} → {new_uri}")
        else:
            task_toml.write_text(new_text)
        rewritten += 1

    suffix = " (dry-run, no writes)" if args.dry_run else ""
    print(f"Rewrote {rewritten} task.toml files{suffix}.")
    if skipped_no_skill:
        print(f"  Skipped {skipped_no_skill}: parent dir didn't match a known skill name.")
    if skipped_no_image:
        print(f"  Skipped {skipped_no_image}: no docker_image field to update.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
