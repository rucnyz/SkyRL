"""Compute GRPO mixed-group ratio for a harbor_pgc training run, using the
SAME fractional reward formula the trainer applies via
``harbor_pgc/llms/skyrl_native_llm.py:_fractional_reward``.

Why a separate script: harbor's ``result.json`` only stores the original
binary reward (0 or 1) from the verifier's reward.txt; our trainer reads
the fractional reward derived from ``verifier/ctrf.json`` at rollout time
and never writes it back. Audit-by-result.json undercounts learning signal.

Usage:
    uv run research/pgc_swe/scripts/mixed_group_analysis.py \
        /scratch/yuzhou/skyrl_runs/codecontest_fracreward_async_v3/trials_run

    # Optional: only count trials whose result.json mtime falls in a window
    --since 2026-05-25T01:30:00 --until 2026-05-25T06:11:00
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# Keep these in sync with harbor_pgc/llms/skyrl_native_llm.py
ALL_PASS_BONUS = 1.2


def fractional_reward(ctrf_path: Path) -> float | None:
    """Same formula as harbor_generator._fractional_reward."""
    try:
        ctrf = json.loads(ctrf_path.read_text())
    except Exception:
        return None
    summary = ctrf.get("results", {}).get("summary", {})
    n = summary.get("tests")
    p = summary.get("passed")
    if n is None or p is None or n <= 0:
        return None
    frac = p / n
    return ALL_PASS_BONUS if frac >= 1.0 else frac


def parse_iso(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trials_dir", help="Path to <run>/trials_run/")
    ap.add_argument("--since", help="ISO timestamp lower bound on result.json mtime")
    ap.add_argument("--until", help="ISO timestamp upper bound on result.json mtime")
    ap.add_argument(
        "--min-group-size",
        type=int,
        default=8,
        help="Require this many samples per task_name to count as a complete GRPO group (default 8)",
    )
    args = ap.parse_args()

    trials_dir = Path(args.trials_dir).resolve()
    if not trials_dir.is_dir():
        print(f"not a dir: {trials_dir}", file=sys.stderr)
        return 2

    since_ts = parse_iso(args.since).timestamp() if args.since else None
    until_ts = parse_iso(args.until).timestamp() if args.until else None

    n_dirs = 0
    n_no_result = 0
    n_no_ctrf = 0
    rows = []  # (task_name, binary_reward, fractional_reward)
    for d in sorted(os.listdir(trials_dir)):
        td = trials_dir / d
        if not td.is_dir():
            continue
        n_dirs += 1
        rp = td / "result.json"
        if not rp.is_file():
            n_no_result += 1
            continue
        if since_ts or until_ts:
            mt = rp.stat().st_mtime
            if since_ts and mt < since_ts:
                continue
            if until_ts and mt > until_ts:
                continue
        try:
            r = json.loads(rp.read_text())
        except Exception:
            continue
        task = r.get("task_name", "?")
        rew_dict = (r.get("verifier_result") or {}).get("rewards") or {}
        bin_rew = rew_dict.get("reward")
        frac = fractional_reward(td / "verifier" / "ctrf.json")
        if frac is None:
            n_no_ctrf += 1
            # Fallback to binary so trial isn't dropped from per-task aggregates.
            frac = float(bin_rew) if bin_rew is not None else None
        rows.append((task, bin_rew, frac))

    print(f"Scanned {n_dirs} trial dirs in {trials_dir}")
    if since_ts or until_ts:
        print(f"  filter: since={args.since} until={args.until}")
    print(f"  with result.json: {len(rows)}")
    print(f"  missing result.json: {n_no_result}")
    print(f"  missing/unparseable ctrf.json: {n_no_ctrf} (fell back to binary)")
    print()

    # ----- Reward distribution under fractional formula -----
    fracs = [r[2] for r in rows if r[2] is not None]
    binaries = [r[1] for r in rows if r[1] is not None]
    if not fracs:
        print("No usable rewards.")
        return 1

    def hist(vals: list[float], bins: list[tuple[float, float]]) -> list[int]:
        return [sum(1 for v in vals if lo <= v < hi) for lo, hi in bins]

    bins = [(0, 1e-9), (1e-9, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0), (1.0 - 1e-9, 1.0 + 1e-9), (1.0 + 1e-9, 2.0)]
    labels = ["= 0", "(0, 0.25)", "[0.25, 0.5)", "[0.5, 0.75)", "[0.75, 1)", "= 1.0", "> 1.0 (bonus)"]
    h = hist(fracs, bins)
    print("Fractional reward histogram (n={}):".format(len(fracs)))
    for lab, c in zip(labels, h):
        pct = 100 * c / len(fracs)
        bar = "#" * int(pct / 2)
        print(f"  {lab:>14}: {c:>6} ({pct:5.1f}%) {bar}")
    print(f"  mean fractional = {sum(fracs)/len(fracs):.4f}")
    if binaries:
        print(f"  mean binary     = {sum(binaries)/len(binaries):.4f}  (what result.json reports)")
    print()

    # ----- GRPO group dynamics -----
    by_task_frac: dict[str, list[float]] = defaultdict(list)
    by_task_bin: dict[str, list[float]] = defaultdict(list)
    for task, b, f in rows:
        if f is not None:
            by_task_frac[task].append(f)
        if b is not None:
            by_task_bin[task].append(b)

    complete_frac_groups = [g for g in by_task_frac.values() if len(g) >= args.min_group_size]
    complete_bin_groups = [g for g in by_task_bin.values() if len(g) >= args.min_group_size]

    def mixed_pct(groups: list[list[float]]) -> tuple[int, int, float]:
        if not groups:
            return 0, 0, 0.0
        mixed = sum(1 for g in groups if len(set(g)) > 1)
        return mixed, len(groups), 100 * mixed / len(groups)

    mf, tf, pf = mixed_pct(complete_frac_groups)
    mb, tb, pb = mixed_pct(complete_bin_groups)

    print(f"GRPO group dynamics (group_size >= {args.min_group_size}):")
    print(f"  fractional reward:  mixed = {mf}/{tf}  ({pf:.1f}%)")
    print(f"  binary    reward:   mixed = {mb}/{tb}  ({pb:.1f}%)")
    if pf > pb + 5:
        print(f"  → fractional gives +{pf-pb:.1f}pp more groups with gradient signal")
    print()

    # ----- Per-task: which group dynamics changed? -----
    flipped = []  # tasks that were all-same in binary but mixed in fractional
    for task, fg in by_task_frac.items():
        if len(fg) < args.min_group_size:
            continue
        bg = by_task_bin.get(task, [])
        if len(set(fg)) > 1 and len(set(bg)) <= 1:
            flipped.append((task, len(fg), fg, bg))
    print(f"Tasks RECOVERED by fractional reward (binary all-same but fractional has variance): {len(flipped)}")
    for task, n, fg, bg in flipped[:5]:
        fg_summary = f"min={min(fg):.2f}, max={max(fg):.2f}, std={statistics.pstdev(fg):.3f}"
        bg_summary = f"all={bg[0]}" if bg else "n/a"
        print(f"  {task}  (n={n})  binary: {bg_summary}  fractional: {fg_summary}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
