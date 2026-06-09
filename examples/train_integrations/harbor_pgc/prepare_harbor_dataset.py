"""
Prepare Harbor task datasets from HuggingFace Hub.

Handles three on-Hub layouts:

  1. Parquet shards with ``path`` + ``task_binary`` columns (e.g.
     ``nvidia/Nemotron-Terminal-Corpus``). Extracted via ``extract_parquet``.

  2. ``.tar.gz`` shards each containing per-task directories (e.g.
     ``nvidia/Nemotron-Terminal-Synthetic-Tasks`` — ``mixed/<skill>.tar.gz``,
     ``easy.tar.gz``, ``medium_shard*.tar.gz``). Extracted via
     ``extract_tarballs``, flattening task dirs into ``output_dir/<task_name>/``.

  3. Loose task directories on the snapshot. Symlinked.

Output directory defaults to ``data/<repo-name>`` next to this script
(i.e. ``examples/train_integrations/harbor_pgc/data/<repo-name>``). Land
the data inside the example dir so reproducers know exactly where it
goes. The ``data/`` subdir is .gitignored.

Usage:

    uv run examples/train_integrations/harbor_pgc/prepare_harbor_dataset.py \\
        --dataset nvidia/Nemotron-Terminal-Synthetic-Tasks

    # Only extract the mixed/ shard (default for run_nemotron_terminal*.sh).
    uv run examples/train_integrations/harbor_pgc/prepare_harbor_dataset.py \\
        --dataset nvidia/Nemotron-Terminal-Synthetic-Tasks --shards mixed
"""

import argparse
import io
import os
import shutil
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path, PurePosixPath

import pyarrow.parquet as pq


def _is_within(base: Path, target: Path) -> bool:
    try:
        return os.path.commonpath([str(base.resolve()), str(target.resolve())]) == str(base.resolve())
    except Exception:
        return False


def _sanitize_tar_member_name(name: str) -> str:
    p = PurePosixPath(name)
    parts = [part for part in p.parts if part not in ("..", ".", "")]
    while parts and parts[0] == "/":
        parts.pop(0)
    return str(PurePosixPath(*parts)) if parts else ""


def _safe_extract_tar(archive_bytes: bytes, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO(archive_bytes)
    with tarfile.open(fileobj=buf, mode="r:*") as tf:
        for member in tf.getmembers():
            member_name = _sanitize_tar_member_name(member.name)
            if not member_name or member_name.endswith("/"):
                (dest_dir / member_name).mkdir(parents=True, exist_ok=True)
                continue
            if ".snapshot" in PurePosixPath(member_name).parts:
                continue
            target = (dest_dir / member_name).resolve()
            if not _is_within(dest_dir, target):
                raise RuntimeError(f"Unsafe path in archive: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if member.isfile():
                with tf.extractfile(member) as src:
                    if src is None:
                        continue
                    with open(target, "wb") as dst:
                        dst.write(src.read())
            elif member.isdir():
                target.mkdir(parents=True, exist_ok=True)


def _extract_one(args: tuple) -> bool:
    """Extract a single task from its tar archive. Runs in a worker process."""
    rel_path, data, output_dir_str = args
    if not isinstance(rel_path, str) or not isinstance(data, (bytes, bytearray, memoryview)):
        return False
    output_dir = Path(output_dir_str)

    safe_rel = PurePosixPath(rel_path)
    parts = [p for p in safe_rel.parts if p not in ("..", "")]
    rel_norm = Path(*parts) if parts else Path("task_unknown")
    target_dir = (output_dir / rel_norm).resolve()

    if not _is_within(output_dir, target_dir):
        return False

    if target_dir.exists() and (target_dir / "instruction.md").exists():
        return True

    try:
        _safe_extract_tar(bytes(data), target_dir)
        # Some tasks reference `COPY files/ /app/` in their Dockerfile but ship
        # an empty environment/files/ — tar archives drop empty dirs, breaking
        # local builds (nitrobox/docker): "/files": not found. Recreate it.
        env_dir = target_dir / "environment"
        if (env_dir / "Dockerfile").exists():
            (env_dir / "files").mkdir(exist_ok=True)
        return True
    except Exception as e:
        print(f"  Warning: Failed to extract {rel_path}: {e}")
        return False


def extract_parquet(parquet_path: Path, output_dir: Path) -> int:
    """Extract tasks from a parquet file with path + task_binary columns."""
    table = pq.read_table(parquet_path)
    path_col = table.column("path").to_pylist()
    data_col = table.column("task_binary").to_pylist()

    output_dir.mkdir(parents=True, exist_ok=True)
    args = [(p, d, str(output_dir)) for p, d in zip(path_col, data_col)]

    with ProcessPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_extract_one, args, chunksize=64))

    return sum(results)


def _shard_matches(rel_path: Path, shards: list[str] | None) -> bool:
    if not shards:
        return True
    rel_str = rel_path.as_posix()
    return any(s in rel_str for s in shards)


def _normalize_parts(name: str) -> tuple[str, ...]:
    return tuple(p for p in PurePosixPath(name).parts if p not in ("", "."))


def extract_tarballs(
    snapshot_dir: Path,
    output_dir: Path,
    shards: list[str] | None = None,
) -> int:
    """Extract ``.tar.gz`` shards into ``output_dir/<task_name>/``.

    Task roots are identified by the presence of ``task.toml`` (every Harbor
    task ships one). The whole task subtree is then materialised under
    ``output_dir/<task_name>/`` regardless of how deep the shard nests it
    (``./<skill>/<task>/`` in ``mixed/`` vs ``./easy_5000/<skill>/<task>/`` in
    ``easy.tar.gz``).

    Idempotent: tasks whose ``instruction.md`` is already on disk are skipped.
    """
    tarballs = sorted(snapshot_dir.glob("**/*.tar.gz"))
    if shards:
        tarballs = [t for t in tarballs if _shard_matches(t.relative_to(snapshot_dir), shards)]
    if not tarballs:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for tb in tarballs:
        rel = tb.relative_to(snapshot_dir)
        print(f"Extracting {rel}...")
        extracted_in_shard = 0
        with tarfile.open(tb, mode="r:*") as tf:
            members = tf.getmembers()
            # Identify task roots by presence of task.toml. Map: full
            # in-tarball task dir path → its members.
            task_root_to_name: dict[tuple[str, ...], str] = {}
            for m in members:
                parts = _normalize_parts(m.name)
                if parts and parts[-1] == "task.toml" and len(parts) >= 2:
                    task_root_to_name[parts[:-1]] = parts[-2]

            if not task_root_to_name:
                continue

            # Bucket every member under its containing task root (if any).
            by_root: dict[tuple[str, ...], list] = {root: [] for root in task_root_to_name}
            for m in members:
                parts = _normalize_parts(m.name)
                for root in task_root_to_name:
                    if len(parts) > len(root) and parts[: len(root)] == root:
                        by_root[root].append(m)
                        break

            for root, members_in_task in by_root.items():
                task_name = task_root_to_name[root]
                target_dir = output_dir / task_name
                if (target_dir / "instruction.md").exists():
                    continue  # already extracted (idempotency)
                target_dir.mkdir(parents=True, exist_ok=True)
                for m in members_in_task:
                    parts = _normalize_parts(m.name)
                    inside = parts[len(root) :]
                    if not inside or ".snapshot" in inside:
                        continue
                    target = (target_dir / Path(*inside)).resolve()
                    if not _is_within(target_dir, target):
                        continue
                    if m.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                        continue
                    if not m.isfile():
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tf.extractfile(m) as src:
                        if src is None:
                            continue
                        with open(target, "wb") as dst:
                            dst.write(src.read())
                extracted_in_shard += 1
        print(f"  extracted {extracted_in_shard} tasks from {rel.name}")
        total += extracted_in_shard
    return total


def prepare(
    dataset_name: str,
    output_dir: str | None = None,
    shards: list[str] | None = None,
) -> str:
    from huggingface_hub import snapshot_download

    repo_name = dataset_name.split("/")[-1] if "/" in dataset_name else dataset_name
    if output_dir is None:
        # Default to ``data/<repo-name>`` colocated with this script so the
        # dataset lives inside the example directory (gitignored).
        output_dir = str(Path(__file__).resolve().parent / "data" / repo_name)
    output_path = Path(os.path.expanduser(output_dir)).resolve()

    print(f"Downloading {dataset_name}...")
    snapshot_dir = Path(snapshot_download(repo_id=dataset_name, repo_type="dataset"))
    print(f"Downloaded to {snapshot_dir}")

    # Find parquet files with path + task_binary columns
    parquets = []
    for f in snapshot_dir.glob("**/*.parquet"):
        try:
            schema = pq.read_schema(f)
            if "path" in schema.names and "task_binary" in schema.names:
                parquets.append(f)
        except Exception as e:
            print(f"  Warning: Could not read schema from {f}: {e}")
            continue

    if not parquets:
        # No parquet — try .tar.gz shards (Nemotron-Terminal-Synthetic-Tasks).
        tarballs = list(snapshot_dir.glob("**/*.tar.gz"))
        if tarballs:
            # Detach any prior symlink at output_path before extracting into it.
            if output_path.is_symlink():
                output_path.unlink()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            total = extract_tarballs(snapshot_dir, output_path, shards=shards)
            print(f"Done! {total} tasks extracted to {output_path}")
            return str(output_path)

        # Last-resort fallback: dataset already has loose task dirs; symlink.
        print("No parquet/tar.gz shards found, symlinking snapshot directly...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.is_symlink():
            output_path.unlink()
        elif output_path.exists():
            shutil.rmtree(output_path)
        output_path.symlink_to(snapshot_dir)
        print(f"Done! Symlinked {output_path} -> {snapshot_dir}")
        return str(output_path)

    total = 0
    for pq_file in parquets:
        print(f"Extracting {pq_file.name}...")
        total += extract_parquet(pq_file, output_path)

    print(f"Done! {total} tasks extracted to {output_path}")
    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Harbor task dataset from HuggingFace Hub")
    parser.add_argument("--dataset", required=True, help="HuggingFace dataset (e.g. nvidia/Nemotron-Terminal-Synthetic-Tasks)")
    parser.add_argument("--output_dir", default=None, help="Output directory (default: data/<repo-name> next to this script)")
    parser.add_argument(
        "--shards",
        nargs="+",
        default=None,
        help="For tar.gz datasets, only extract tarballs whose relative path "
        "contains one of these substrings (e.g. ``mixed``, ``easy``, ``medium``).",
    )
    args = parser.parse_args()
    prepare(args.dataset, args.output_dir, shards=args.shards)
