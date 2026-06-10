# Running harbor_pgc on nitrobox (local sandboxes, no cloud account)

The `nitrobox` branch swaps the harbor sandbox backend from E2B to
[nitrobox](https://github.com/opensage-agent/nitrobox) — local Linux
user-namespace sandboxes via the [rucnyz/harbor](https://github.com/rucnyz/harbor)
fork. No E2B/Daytona/Modal account or API key is needed; task images are built
locally through nitrobox's embedded buildkit (per-skill base layers are shared
via the layer cache, so after the first cold build a sandbox boots in seconds).

Everything (harbor fork, nitrobox) is pinned in `pyproject.toml`/`uv.lock` and
installed automatically by `uv run` — cloning this repo on the `nitrobox`
branch is all the code you need.

Verified 2026-06-09/10 on 1 node (B300s, Qwen3.5-9B): sync 2 steps + fully
async 2+2 steps with in-flight weight updates (44 trials mid-flight during a
2.4s weight swap, all resumed cleanly).

## Prerequisites

1. **Linux with unprivileged user namespaces enabled.** Ubuntu 23.10+
   restricts them by default; you must (once, as root):

   ```bash
   sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0
   echo "kernel.apparmor_restrict_unprivileged_userns = 0" | sudo tee /etc/sysctl.d/99-nitrobox-userns.conf
   ```

   Failure signatures if this is missing:
   - `RuntimeError: buildkitd failed to start within 30s`
   - `SandboxKernelError: ... write failed /proc/self/uid_map: Operation not permitted`

   No Docker daemon and no other root setup is required.

2. **GPUs**: 2 GPUs for the sync smoke run (Qwen3.5-9B colocated), 4 for the
   fully-async run (2 FSDP train + 2 vLLM engines). Plenty of CPU cores/RAM
   helps — agent sandboxes and verifiers run locally.

3. **uv** installed; CUDA toolkit at `/usr/local/cuda` (scripts export
   `CUDA_HOME` themselves).

4. **wandb** (optional): scripts default to `trainer.logger=wandb`. Either
   export `WANDB_API_KEY` (or point `PGC_ENV_FILE` at an env file containing
   it), or append `trainer.logger=console` to the launch command.

## Dataset

```bash
cd <repo root>
uv run examples/train_integrations/harbor_pgc/prepare_harbor_dataset.py \
    --dataset nvidia/Nemotron-Terminal-Synthetic-Tasks
```

This extracts ~6k tasks into
`examples/train_integrations/harbor_pgc/data/Nemotron-Terminal-Synthetic-Tasks/`
(and recreates the empty `environment/files/` dirs that tar drops — required
for local `COPY files/` builds).

The smoke scripts use a 16-task subset. Create it with symlinks:

```bash
D=examples/train_integrations/harbor_pgc/data
mkdir -p $D/Nemotron-Terminal-Synthetic-Tasks_smoke16
for t in data_processing_task_000{1..8} data_science_task_000{1..8}; do
  ln -s "$(pwd)/$D/Nemotron-Terminal-Synthetic-Tasks/$t" \
        "$D/Nemotron-Terminal-Synthetic-Tasks_smoke16/$t"
done
```

## Run

Sync GRPO smoke (2 GPUs, 8 prompts x 4 samples, 2 steps):

```bash
bash examples/train_integrations/harbor_pgc/run_nemotron_terminal_nitrobox.sh
```

Fully async (in-flight weight update / multi-turn partial rollout; 4 GPUs):

```bash
bash examples/train_integrations/harbor_pgc/run_nemotron_terminal_nitrobox_fully_async.sh
```

Useful overrides (env vars consumed by the scripts):

```bash
CUDA_VISIBLE_DEVICES=4,5 RUN_NAME=my_smoke \
  bash examples/train_integrations/harbor_pgc/run_nemotron_terminal_nitrobox.sh \
  trainer.logger=console            # extra args are passed through to hydra
```

To train on the full dataset instead of smoke16, override the data paths
(must be **absolute** — relative paths resolve inside ray's packaged
working_dir, where `data/` does not exist):

```bash
bash examples/train_integrations/harbor_pgc/run_nemotron_terminal_nitrobox_fully_async.sh \
  "data.train_data=[$(pwd)/examples/train_integrations/harbor_pgc/data/Nemotron-Terminal-Synthetic-Tasks]" \
  "data.val_data=[$(pwd)/examples/train_integrations/harbor_pgc/data/Nemotron-Terminal-Synthetic-Tasks]"
```

## What to expect

- First run per skill: cold base-image build (apt+pip), minutes per skill —
  `environment_build_timeout_multiplier: 5` in
  `harbor_trial_config/default.yaml` covers this. Cached rebuilds: seconds.
- Sync smoke: generate ~13 min (32 trials), `fwd_logprobs` ~6 min,
  `policy_train` ~23 min per step on 2 GPUs.
- Fully async: trainer steps whenever 8 groups are buffered; generation keeps
  running during training, and each step ends with a ~2s in-flight weight
  swap (`pause_generation` -> NCCL broadcast -> `resume_generation`; frozen
  requests resume on the new weights).
- Trial artifacts land in `$HOME/skyrl_runs/<RUN_NAME>/trials_run/<trial>/`
  (agent logs, `verifier/ctrf.json` for fractional reward, `result.json`).
- buildkit state/cache: `~/.local/share/nitrobox/buildkit` (shared across
  runs; safe to delete when no run is active).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `buildkitd failed to start within 30s` or `SandboxKernelError ... uid_map` | userns restricted — apply the sysctl above |
| `dataset should be at least as large as train_batch_size ... got size 0` | data path was relative — use absolute paths |
| `failed to calculate checksum of ref ...: "/files": not found` | dataset extracted with an old prepare script — `for d in data/<dataset>/*/environment; do mkdir -p $d/files; done` |
| Transient `JSONDecodeError` (buildkit) / `AgentSetupTimeoutError` during the first minutes | cold-build storm; per-trial retries absorb these, they disappear once base images are cached |
| Orphan `sleep infinity` processes after killing a run | nitrobox network-namespace holders — `pkill -u $USER -f '^sleep infinity'` |
