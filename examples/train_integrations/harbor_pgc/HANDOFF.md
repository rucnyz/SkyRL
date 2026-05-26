# Handoff — harbor_pgc PGC training (2026-05-24)

Snapshot of what's running, why it's set up this way, and how to take over.
Transient by design: delete once you've cloned in and confirmed your own
run works.

## What's running

**A 6-GPU fully-async GRPO training run** on the dev box, started 20:07 UTC
2026-05-24 via:

```bash
bash examples/train_integrations/harbor_pgc/run_nemotron_terminal_fully_async.sh \
  > /scratch/yuzhou/projects/SkyRL/.pgc-logs/async_v2.log 2>&1 &
```

| Knob | Value |
|---|---|
| Model | `Qwen/Qwen3.5-9B` (`language_model_only=true`) |
| Dataset | `nvidia/Nemotron-Terminal-Synthetic-Tasks` mixed shard (5984 prompts, 6 skills) |
| Algorithm | GRPO, token_mean loss, no KL, TIS (token, clip_high=2.0) |
| Reward | **Fractional `passed/tests` + 1.2 bonus on all-pass** — see `harbor_generator.py:_fractional_reward` |
| Trainer | `FullyAsyncRayPPOTrainer`, max_staleness_steps=4, num_parallel_generation_workers=128 |
| Batch | `train_batch_size=64`, `n_samples_per_prompt=8`, `policy_mini_batch_size=64` |
| Epochs | 3 (279 total training steps, 93 per epoch) |
| LR | 1e-6 (constant) |
| Max gen length | 16384 tokens; max seq 262144 |
| GPUs | `CUDA_VISIBLE_DEVICES=0,1,2,3,5,6` (skip 4/7, occupied by another user's vllm-serve) |
| Layout | 2 FSDP train (GPU 5,6) + 4 vLLM inference engines (GPU 0,1,2,3, TP=1) |
| e2b concurrency | 64 simultaneous sandboxes, 5 trajectories/sec ramp |
| ckpt cadence | every 5 steps, keep last 5 |

**Identifiers**:

- Process: `pid 2902782` (`ray::skyrl_entrypoint`), parent shell `pid 2883485`
- Log: `/scratch/yuzhou/projects/SkyRL/.pgc-logs/async_v2.log` (already 100MB+)
- Output root: `/scratch/yuzhou/skyrl_runs/codecontest_fracreward_async_v2/`
  - `trials_run/` — per-trial directories (result.json + agent/trajectory.json + verifier/ctrf.json)
  - `ckpts/global_step_N/` — FSDP checkpoints (none saved yet — see below)
  - `exports/` — HF-format snapshots
- WandB run: project `harbor`, run name `codecontest_fracreward_async_v2`, owner `rucnyz`

## Where we are right now (as of 2026-05-24 21:24 UTC)

- **Generation Buffer Progress**: still filling — needs 64 prompt-groups with 8 completed samples each before the first GRPO update fires. ~370 trials completed of the ~512 needed; ETA to first `grad_norm` log roughly another 30–60 minutes from now.
- **No `Saved checkpoint`, no `grad_norm` yet** — first update hasn't happened.
- e2b health: 64 sandboxes active continuously, occasional 65 (DELETE/CREATE race window), 2 `_maybe_download_logs` ERRORs in ~70 minutes (vs >100 in the failed v1 run over similar time — healthy).
- Throughput so far: ~5.5 trials/min steady-state, matches the proven sync run.

## Why this layout (avoid the v1 mistake)

There's a tight coupling between e2b sandbox concurrency and vLLM inference
capacity that the upstream fully-async defaults get wrong for our model.

Earlier today I ran with:
- `MAX_CONCURRENCY=80`, `NUM_INFERENCE_ENGINES=2`
- → 40 agents per vLLM engine
- → LLM responses queue, agents idle 5–20 minutes per turn, e2b's per-sandbox
  HTTP/2 connections get reaped server-side
- → `harbor.agents.terminus_2.tmux_session.capture_pane` times out under
  tenacity retry, `harbor.environments.e2b:stop` starts failing en masse
- → no `result.json` written even though sandboxes "exist", buffer never fills

The proven-healthy sync run that lived for 25 hours uses 64 concurrency with 4
engines colocated on 4 GPUs = **16 agents per engine**. The current v2 launcher
reproduces that ratio with the GPUs split (4 inference + 2 FSDP train = 6 GPUs
total).

Don't push concurrency above the 16:1 ratio. If you need more parallelism,
add inference engines first, not sandbox concurrency.

## Code on GitHub

Branch `pgc-swe` of `https://github.com/rucnyz/SkyRL.git`, head commit
`bad36b23` as of this writing. Five commits today:

```
bad36b23  drop legacy prebuild_e2b_templates.py; align run_harbor_gen GPU default
25f12b02  drop deprecated smoke launcher; README documents fractional reward + ratio guidance
24888631  run_nemotron_terminal_fully_async.sh: 6-GPU 2+4 layout, batch=64, MAX_CONCURRENCY=64
0ca0da0c  run_nemotron_terminal.sh: batch_size 32->64, MAX_CONCURRENCY 64->80, resume_mode=none
94b7c2da  fractional pass@N reward (passed/tests w/ all-pass bonus 1.2)
```

The NeMo-RL repo at `/scratch/yuzhou/projects/RL/` has 9 unpushed commits but
none of them affect the PGC path — async training runs entirely on SkyRL. You
do not need that repo to reproduce, you only need an `E2B_API_KEY` and a
`WANDB_API_KEY`.

## Reproducing on a different box

```bash
# 1. Clone our SkyRL fork
git clone https://github.com/rucnyz/SkyRL.git
cd SkyRL && git checkout pgc-swe

# 2. Credentials. Either export directly:
export E2B_API_KEY=...                # from https://e2b.dev/dashboard, paid account (100 sandbox cap)
export WANDB_API_KEY=...              # optional, console logger if unset
# Or drop them into a file and point at it:
export PGC_ENV_FILE=/path/to/your/.env

# 3. Prepare dataset (~few min, ~3GB extracted)
uv run examples/train_integrations/harbor_pgc/prepare_harbor_dataset.py \
    --dataset nvidia/Nemotron-Terminal-Synthetic-Tasks --shards mixed

# 4. ghcr images: by default the prepared task.toml files point at
#    ghcr.io/rucnyz/nemotron-<skill>:1.0 which are public. If those have
#    moved or you want to host your own, rebuild + push and rewrite:
bash examples/train_integrations/harbor_pgc/scripts/build_skill_images.sh
uv run -m examples.train_integrations.harbor_pgc.scripts.rewrite_task_dockerimage \
    examples/train_integrations/harbor_pgc/data/Nemotron-Terminal-Synthetic-Tasks

# 5. Launch
bash examples/train_integrations/harbor_pgc/run_nemotron_terminal_fully_async.sh
```

If your box has a different free-GPU set, override before launch:
`CUDA_VISIBLE_DEVICES=... bash run_nemotron_terminal_fully_async.sh`.
Keep 6 GPUs total or adjust `NUM_INFERENCE_ENGINES` + `NUM_POLICY_GPUS` in
the script accordingly; the 16:1 agents/engine ratio should hold.

## Monitoring this run

- WandB: https://wandb.ai/rucnyz/harbor/runs (run named `codecontest_fracreward_async_v2`)
- Log tail: `tail -F /scratch/yuzhou/projects/SkyRL/.pgc-logs/async_v2.log`
- e2b sandbox count:
  ```bash
  curl -sS -H "X-API-KEY: $E2B_API_KEY" https://api.e2b.app/sandboxes \
    | python3 -c "import sys,json; print(len(json.load(sys.stdin)))"
  ```
- Trial throughput:
  ```bash
  find /scratch/yuzhou/skyrl_runs/codecontest_fracreward_async_v2/trials_run \
      -maxdepth 2 -name result.json | wc -l
  ```

## Old runs preserved on disk

- `/scratch/yuzhou/skyrl_runs/codecontest/` (650 GB) — the 25-hour sync run.
  Has ckpts at global_step_{5,10,15,20} and 7192 trial dirs with full
  trajectories. Kept for reference / pre-fractional-reward comparison.
- `/scratch/yuzhou/skyrl_runs/codecontest_fracreward_async_v1/` — *deleted*
  (the failed 80-conc/2-engine run).
- `/scratch/yuzhou/skyrl_runs/codecontest-smoke/` — *deleted* (old smoke
  testing).

## Open questions for next-shift

1. **Does fractional reward materially help?** Compare `wandb.runs[codecontest_fracreward_async_v2].history` vs `wandb.runs[codecontest].history` once the new run gets past step 5. Hypothesis: % of mixed GRPO groups jumps from ~34% to ~90%; reward variance drops; effective batch size goes up.
2. **Is async actually faster end-to-end than sync?** Sync took ~3.5h per step (71min gen + ~1h fwd + 30–60min policy_train), async should hide the gen-train serial cost. Need to see one full step cycle (gen + train) under fully-async to confirm.
3. **2 train GPUs vs 4** — current v2 uses 2 train GPUs. If train_critic_and_policy becomes the wall-time bottleneck, consider 4 train + 2 inference (back to v1's failed shape) but only if vLLM throughput per engine is high enough that 2 engines handle 64 agents — unlikely.

## Killing it cleanly

```bash
# 1. SIGTERM the ray actor + parent. Agent loop will catch and DELETE its e2b sandbox.
kill -TERM 2902782 2883485
sleep 10
# 2. Force any survivors
pkill -KILL -f 'skyrl_entrypoint|main_harbor_fully_async'
# 3. Verify e2b is clean — if any sandboxes survived, list and delete:
curl -sS -H "X-API-KEY: $E2B_API_KEY" https://api.e2b.app/sandboxes \
  | python3 -c "import sys,json; [print(s['sandboxID']) for s in json.load(sys.stdin)]" \
  | xargs -I{} -P 16 curl -sS -X DELETE -H "X-API-KEY: $E2B_API_KEY" "https://api.e2b.app/sandboxes/{}"
```

## Things that bit us today, written down so they don't bite again

- **e2b account cap is 100, not "soft 100"** — at MAX_CONCURRENCY=80 the
  zombie inflation pushed instantaneous count to 81+ multiple times. Stay
  ≤ 80 with the zombie reaper enabled, or refactor the reaper for tighter
  bookkeeping if you need to push higher.
- **Sandbox leaks past harbor's stop()** — harbor's
  ``E2BEnvironment.stop()`` catches any exception from ``self._sandbox.kill()``,
  logs an ERROR, and proceeds to ``self._sandbox = None``. If the SDK kill
  failed (transient e2b 5xx), the sandbox stays alive on the account quota
  until e2b's own inactivity_timeout reaps it (observed: many hours).
  Mitigation in ``SharedTemplateE2BEnvironment``:
    1. A process-wide ``_LIVE_ENVIRONMENT_NAMES`` set tracks active trials.
    2. ``_create_sandbox`` stamps ``owner_pid`` into metadata.
    3. ``stop()`` always deregisters + fires a backup REST DELETE.
    4. A background ``_owner_reaper_loop`` (60s interval) lists all e2b
       sandboxes; any with our PID whose ``environment_name`` is not in the
       live set gets DELETE'd. Foreign-PID and pre-fix (no-PID) sandboxes
       are left alone, so concurrent runs on the same E2B account are safe.
  See ``examples/train_integrations/harbor_pgc/tests/test_owner_reaper.py``
  for the mock-based correctness tests.
- **vLLM 2 engines is not enough for 64+ concurrent agents** — see "Why this
  layout" above. Symptom is no `result.json` getting written even though
  sandboxes look alive.
- **harbor_generator.py imports are loaded once at Ray actor startup** —
  editing the file while training is running has no effect on the running
  job; you need a fresh launch.
- **`trainer.resume_mode=latest` + same run_name = resume**; `none` + new
  run_name = fresh start. Don't `rm -rf ckpts/` to "reset" — change run_name
  instead so the old ckpts stay for comparison.
- **Sync vs async wall-time accounting** — sync was 3.5h/step end-to-end
  with 79 days/epoch ETA. Async should be max(gen, train) per step. If
  async beats sync, gen and train are running concurrently as intended;
  if it doesn't, FullyAsyncRayPPOTrainer isn't actually overlapping them
  and you're paying the GPU split cost for nothing.
