"""Pre-build the E2B template alias each Harbor CodeContests task uses.

Why: harbor's E2BEnvironment lazily builds a per-task template alias on the
first `trial.run()`. When SkyRL training fires N parallel rollouts pointing
at distinct task aliases that all happen to share the same Dockerfile
contents, E2B's per-account build rate-limiter cancels most of them and
leaves zombie aliases that subsequently 404 on `sandbox.create()` (see
~/.claude/projects/.../memory/e2b_zombie_aliases.md). Pre-building each
alias sequentially side-steps the race entirely.

Usage:
    uv run --isolated --extra harbor --with "harbor[e2b]" \\
        -m examples.train_integrations.harbor_pgc.prebuild_e2b_templates \\
        [DATASET_DIR]

DATASET_DIR defaults to ~/data/harbor/CodeContests. CPU/MEM defaults to
the same values our harbor_trial_config/default.yaml overrides to (1 / 1024).

For each task dir we compute the same template alias name as
harbor.environments.e2b.E2BEnvironment.__init__ does:
    "{env_name}__{sha256(env_dir)[:8]}"  (with / and . sanitised)
and call AsyncTemplate.build(skip_cache=True) so any pre-existing zombie
alias gets replaced.

Sequential by default (concurrency=1) — the whole point is to avoid E2B's
parallel-build rate limiter. Raise `--concurrency N` only if you've
verified your E2B plan allows it.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dirhash import dirhash
from e2b import AsyncTemplate, Template


def _alias_for(task_dir: Path, env_dir: Path) -> str:
    """Mirror E2BEnvironment.__init__ alias derivation byte-for-byte."""
    h = dirhash(str(env_dir), "sha256")[:8]
    return f"{task_dir.name}__{h}".replace("/", "__").replace(".", "-")


async def _build_one(
    task_dir: Path,
    cpu_count: int,
    memory_mb: int,
    skip_cache: bool,
    sem: asyncio.Semaphore,
    counter: dict,
) -> None:
    env_dir = task_dir / "environment"
    dockerfile = env_dir / "Dockerfile"
    if not dockerfile.exists():
        return
    alias = _alias_for(task_dir, env_dir)
    async with sem:
        idx = counter["done"] + counter["fail"] + 1
        total = counter["total"]
        print(f"[{idx}/{total}] {alias} ...", flush=True)
        template = Template(file_context_path=str(env_dir)).from_dockerfile(
            dockerfile_content_or_path=str(dockerfile),
        )
        try:
            await AsyncTemplate.build(
                template=template,
                alias=alias,
                cpu_count=cpu_count,
                memory_mb=memory_mb,
                skip_cache=skip_cache,
            )
            counter["done"] += 1
            print(f"    ✓ {alias}", flush=True)
        except Exception as e:  # noqa: BLE001
            counter["fail"] += 1
            print(f"    ✗ {alias}: {type(e).__name__}: {e}", flush=True)


async def _main(args: argparse.Namespace) -> int:
    dataset_dir = Path(args.dataset_dir).expanduser()
    if not dataset_dir.is_dir():
        print(f"DATASET_DIR not a directory: {dataset_dir}", file=sys.stderr)
        return 1

    task_dirs = sorted(
        d for d in dataset_dir.iterdir()
        if d.is_dir() and (d / "environment" / "Dockerfile").exists()
    )
    if args.limit:
        task_dirs = task_dirs[: args.limit]
    counter = {"total": len(task_dirs), "done": 0, "fail": 0}
    print(f"Pre-building {counter['total']} E2B templates from {dataset_dir} "
          f"(cpu={args.cpu_count}, mem={args.memory_mb}MB, "
          f"concurrency={args.concurrency}, skip_cache={args.skip_cache})",
          flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    await asyncio.gather(*(
        _build_one(t, args.cpu_count, args.memory_mb, args.skip_cache, sem, counter)
        for t in task_dirs
    ))

    print(f"\nDone. {counter['done']}/{counter['total']} succeeded, "
          f"{counter['fail']} failed.", flush=True)
    return 0 if counter["fail"] == 0 else 2


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "dataset_dir",
        nargs="?",
        default=os.path.expanduser("~/data/harbor/CodeContests"),
        help="Directory containing one subdir per task (each with environment/Dockerfile).",
    )
    p.add_argument("--cpu-count", type=int, default=1)
    p.add_argument("--memory-mb", type=int, default=1024)
    p.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Parallel template builds. Default 1 (sequential). "
             "Raise only if your E2B plan tolerates it.",
    )
    p.add_argument(
        "--skip-cache",
        action="store_true",
        default=True,
        help="Force a full rebuild even if e2b reports an existing alias. "
             "Recommended (zombie aliases survive failed builds).",
    )
    p.add_argument(
        "--no-skip-cache",
        action="store_false",
        dest="skip_cache",
        help="Trust e2b's existing-alias cache (fast but won't recover zombies).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only pre-build the first N tasks (for smoke / sanity).",
    )
    args = p.parse_args()
    sys.exit(asyncio.run(_main(args)))


if __name__ == "__main__":
    main()
