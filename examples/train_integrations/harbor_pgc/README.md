## Harbor Integration — PGC variant

Fork of `examples/train_integrations/harbor/` for the PGC paper experiments.
RL training with [Harbor](https://github.com/laude-institute/harbor) as
environment + reward source, agent rollouts inside an [E2B](https://e2b.dev/)
cloud sandbox, `Qwen/Qwen3.5-9B` policy, dataset
[`nvidia/Nemotron-Terminal-Synthetic-Tasks`](https://huggingface.co/datasets/nvidia/Nemotron-Terminal-Synthetic-Tasks).

### How this fork differs from upstream

| | upstream `harbor/` | this `harbor_pgc/` |
|---|---|---|
| Sandbox | daytona / modal | **e2b** |
| Dataset | CodeContests | **nvidia/Nemotron-Terminal-Synthetic-Tasks** |
| Model | Qwen3-8B (8K ctx) | Qwen3.5-9B (256K ctx) |
| Inference path | SkyRL legacy HTTP endpoint | upstream-default vllm-router (`_SKYRL_USE_NEW_INFERENCE=1`) |
| LLM client | LiteLLM → `/v1/chat/completions` | `SkyRLNativeLLM` → `/skyrl/v1/generate` |
| Agent | stock `Terminus2` | `SkyRLTerminus2` via `agent.import_path` |
| Environment | stock `E2BEnvironment` | `SharedTemplateE2BEnvironment` via `environment.import_path` |

Two SkyRL-side extensions carry their weight:

**`SkyRLNativeLLM`** (`llms/skyrl_native_llm.py`) — vllm-router's
OpenAI-compatible `/v1/chat/completions` route silently drops the vllm
extension fields (`prompt_token_ids`, `completion_token_ids`, `logprobs`)
that step-wise RL training needs. We instead POST pre-tokenised prompts to
the router's `/skyrl/v1/generate` endpoint, which preserves them.
Chat-template application moves client-side.

**`SharedTemplateE2BEnvironment`** (`environments/skyrl_e2b.py`) — every task
in Nemotron's `<skill>.tar.gz` ships an *identical* per-skill Dockerfile
plus a *unique* `environment/files/` baked in via `COPY files/ /app/`.
Building one e2b template per task would be 5984 builds per dataset run. We
instead pre-build 11 skill-base images on ghcr.io (one per skill, with
`COPY files/` stripped), rewrite every `task.toml` to point at those, and
upload per-task `files/` to `/app/` at sandbox start. One e2b template
alias per skill rather than per task. A per-alias `asyncio.Lock`
serialises concurrent template builds so the first trial through wins
the build and the rest cache-hit.

**Fractional reward** (`harbor_generator.py:_fractional_reward`) — harbor's
default reward is binary (`reward.txt` ∈ {0, 1}: pytest all-passes or
not). With a typical task running 5-8 tests, partial-credit attempts
(e.g. 5/6 passing) get the same 0 reward as no-credit attempts, and GRPO
groups end up dominated by all-zero outcomes that provide no gradient.
We instead read pytest's structured `ctrf.json` and return
`passed/tests`, with an `ALL_PASS_BONUS=1.2` bump when every test
passes so the policy still strictly prefers full success over partial.
Falls back to the binary reward.txt path if ctrf.json is missing.

### Structure

```
harbor_pgc/
  agents/skyrl_terminus_2.py       Terminus-2 subclass that dispatches
                                   llm_backend="skyrl" to SkyRLNativeLLM,
                                   loaded via harbor's agent.import_path.
  llms/skyrl_native_llm.py         BaseLLM subclass posting token_ids to
                                   {proxy_url}/skyrl/v1/generate.
  environments/skyrl_e2b.py        SharedTemplateE2BEnvironment.
  harbor_generator.py              Wires agent.import_path +
                                   environment.import_path + proxy_url.
  dataset.py                       HarborTaskDataset: walks task dirs.
  prepare_harbor_dataset.py        Downloads + extracts a HuggingFace dataset
                                   into data/<repo-name>/ (gitignored).
  scripts/
    build_skill_images.sh          Build + push 11 ghcr.io skill-base images.
                                   One-time per dataset bump.
    rewrite_task_dockerimage.py    Rewrite task.toml docker_image fields to
                                   point at our ghcr images.
  harbor_trial_config/default.yaml Harbor TrialConfig (e2b sandbox + Qwen3.5-9B).
  entrypoints/
    main_harbor.py                 GRPO training entrypoint (sync).
    main_harbor_fully_async.py     Fully-async GRPO entrypoint.
    main_harbor_generate.py        Generation-only sanity entrypoint.
  data/                            (gitignored) prepared task dirs land here.
  run_nemotron_terminal.sh         Full training (sync GRPO).
  run_nemotron_terminal_fully_async.sh
                                   Full training (fully-async GRPO).
  run_harbor_gen.sh                Generation-only sanity launcher.
```

### Running the Nemotron-Terminal training (current setup)

All paths are relative to the SkyRL repo root.

```bash
# 1. Credentials. The dev box's $PGC_ENV_FILE sources both; otherwise set
#    them by hand.
export E2B_API_KEY=...
export WANDB_API_KEY=...       # optional; console logger if unset

# 2. Prepare the dataset (~few min). The --shards mixed restriction extracts
#    only the 6-skill / 5984-task mixed subset that the training scripts read.
#    Drop the flag to extract all 248k tasks across mixed+easy+medium.
uv run examples/train_integrations/harbor_pgc/prepare_harbor_dataset.py \
    --dataset nvidia/Nemotron-Terminal-Synthetic-Tasks --shards mixed

# 3. One-time: build + push the 11 ghcr.io skill-base images. Skip if you
#    are reusing an upstream owner's images (default points at
#    ghcr.io/rucnyz/nemotron-<skill>:1.0). Requires docker + a GitHub PAT
#    with write:packages (gh auth token). After push, flip each package
#    to public at https://github.com/users/<owner>?tab=packages.
bash examples/train_integrations/harbor_pgc/scripts/build_skill_images.sh

# 4. Rewrite every task.toml's docker_image field to the ghcr URIs.
uv run -m examples.train_integrations.harbor_pgc.scripts.rewrite_task_dockerimage \
    examples/train_integrations/harbor_pgc/data/Nemotron-Terminal-Synthetic-Tasks

# 5. Full training. Sync defaults to 4 GPUs (colocated FSDP + vLLM);
#    fully-async defaults to 6 GPUs (2 FSDP train + 4 vLLM inference).
bash examples/train_integrations/harbor_pgc/run_nemotron_terminal.sh
#   or fully-async:
bash examples/train_integrations/harbor_pgc/run_nemotron_terminal_fully_async.sh
```

### Concurrency & vLLM ratio

`max_concurrency` (parallel e2b sandboxes) and `num_inference_engines`
(vLLM replicas) need to stay roughly in step. The 16:1 agents-per-engine
ratio the sync launcher uses (64 conc / 4 engines) has been stable for
multi-day runs; bumping conc to 80 with only 2 engines (the upstream
fully-async default) tips into a death spiral where LLM responses queue,
e2b sandboxes idle past their connection-reuse window, `tmux_session.
capture_pane` calls time out under tenacity retry, and `harbor.environ
ments.e2b:stop` starts failing en masse. The fully-async launcher's
defaults (6 GPUs, 4 engines, 64 conc) keep the same 16:1 ratio.

### Dataset variants

`nvidia/Nemotron-Terminal-Synthetic-Tasks` ships three independent shards
under `skill_based/`. Same task format and same per-skill Dockerfile across
all of them, so the 11 ghcr base images cover everything.

| Shard | Tasks | Skills | Notes |
|---|---|---|---|
| `mixed/*.tar.gz` | **5,984** | 6 (data_processing, data_science, debugging, file_operations, scientific_computing, security) | **Default — what `run_nemotron_terminal*.sh` consume.** |
| `easy.tar.gz` | 44,969 | 9 (+ data_querying, software_engineering, dependency_management) | Skill-ablation scale. |
| `medium_shard1.tar.gz` + `medium_shard2.tar.gz` | 203,749 | 11 (+ model_training, system_administration) | Full bulk synth pool. |

Sister repo `nvidia/Nemotron-Terminal-Corpus` packages the same
synthetic_tasks/ in parquet form plus `dataset_adapters/{code,math,swe}.parquet`
(swe is real SWE-bench bridged into the harbor format).

Other harbor-format public datasets:
- `harborframework/terminal-bench-2.0` — 89 hand-written; official tbench eval.
- `zai-org/terminal-bench-2-verified` — same 89 with instruction/env fixes.
- `laude-institute/sandboxes-tasks` — 94 hand-written from harbor's authors.

### Box-specific notes (dev box)

- `.python-version` pins CPython 3.12 — SkyRL's pinned wheel URLs use
  `python_version == '3.12'` markers; system default 3.13 falls back to
  source builds that then trip torch's nvcc version check.
- `run_nemotron_terminal.sh` defaults to `CUDA_VISIBLE_DEVICES=0,1,2,3`
  (4-GPU colocated sync) and `run_nemotron_terminal_fully_async.sh` to
  `0,1,2,3,5,6` (6 GPUs: 2 FSDP train on 5,6 + 4 vLLM inference on 0–3),
  both avoiding GPU 4 / 7 where another user's stale `VLLM::EngineCore`
  is often sitting.
- B300 GPUs (cc 10.3) handle Qwen3.5-9B at 256K context comfortably.
