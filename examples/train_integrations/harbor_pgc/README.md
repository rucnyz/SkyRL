## Harbor Integration — PGC variant

Fork of `examples/train_integrations/harbor/` used by the PGC paper
experiments. RL training with [Harbor](https://github.com/laude-institute/harbor)
as the environment + reward source, agent rollouts running inside an
[E2B](https://e2b.dev/) cloud sandbox, and a `Qwen/Qwen3.5-9B` policy.

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

**`SkyRLNativeLLM`** — vllm-router's OpenAI-compatible `/v1/chat/completions`
route silently drops the vllm extension fields (`prompt_token_ids`,
`completion_token_ids`, `logprobs`) that step-wise RL training needs.
We instead POST pre-tokenised prompts to the router's `/skyrl/v1/generate`
endpoint, which preserves them. Chat-template application moves
client-side. See `llms/skyrl_native_llm.py`.

**`SharedTemplateE2BEnvironment`** — every task in Nemotron's
`<skill>.tar.gz` ships an *identical* per-skill Dockerfile + a
*unique* `environment/files/` baked in via `COPY files/ /app/`. Building
one e2b template per task would be 5984 builds per dataset. We instead
push 11 skill-base images to ghcr.io once (`scripts/build_skill_images.sh`),
rewrite every `task.toml` to point at those images
(`scripts/rewrite_task_dockerimage.py`), and the subclass uploads
`environment/files/` into `/app/` at sandbox start. One e2b template
alias per skill instead of per task. See `environments/skyrl_e2b.py`.

### Structure

```
harbor_pgc/
  agents/skyrl_terminus_2.py       Terminus-2 subclass; overrides _init_llm
                                   to dispatch llm_backend="skyrl" to our
                                   SkyRLNativeLLM (loaded via harbor's
                                   official agent.import_path extension
                                   point — no monkey-patching).
  llms/skyrl_native_llm.py         BaseLLM subclass: client-side chat
                                   template + POST token_ids to
                                   {proxy_url}/skyrl/v1/generate.
  environments/skyrl_e2b.py        SharedTemplateE2BEnvironment: one e2b
                                   template per docker_image; uploads
                                   per-task environment/files/ into /app/
                                   at sandbox start.
  harbor_generator.py              HarborGenerator: reads proxy_url off the
                                   RemoteInferenceClient at runtime, wires
                                   agent.import_path + environment.import_path
                                   so harbor instantiates our subclasses.
  dataset.py                       HarborTaskDataset: loads task directory
                                   paths from a Harbor-style dataset dump.
  prepare_harbor_dataset.py        Downloads + extracts a HuggingFace dataset
                                   into ``data/<repo-name>/`` next to this
                                   script. ``data/`` is .gitignored.
  prebuild_e2b_templates.py        (legacy fallback) per-task template
                                   pre-builder. Superseded by
                                   build_skill_images.sh + rewrite_task_dockerimage.py
                                   for Nemotron-style datasets; still useful
                                   for ad-hoc tasks with unique Dockerfiles.
  scripts/
    build_skill_images.sh          Build + push the 11 skill-base images to
                                   ghcr.io/<owner>/nemotron-<skill>:<tag>.
                                   Strips `COPY files/` so the image is
                                   task-agnostic. Run once per dataset bump.
    rewrite_task_dockerimage.py    Rewrite every task.toml's docker_image to
                                   point at the ghcr images above.
  harbor_trial_config/default.yaml Harbor TrialConfig template (e2b sandbox,
                                   Qwen3.5-9B model_info).
  entrypoints/
    main_harbor.py                 Full GRPO training entrypoint.
    main_harbor_fully_async.py     Fully-async training entrypoint.
    main_harbor_generate.py        Generation-only sanity entrypoint.
  data/                            (gitignored) prepared task directories
                                   land here, e.g.
                                   data/Nemotron-Terminal-Synthetic-Tasks/.
  run_nemotron_terminal.sh         Nemotron-Terminal training (sync).
  run_nemotron_terminal_fully_async.sh
                                   Nemotron-Terminal training (fully async).
  run_nemotron_terminal_smoke.sh   Tiny-subset smoke training (validates
                                   FSDP + weight sync end-to-end in ~1 hour).
  run_harbor_gen.sh                Generation-only sanity launcher.
```

### Quick Start

```bash
cd SkyRL

# 1. Credentials. PGC_ENV_FILE on the dev box sets both E2B_API_KEY and
#    WANDB_API_KEY; set them by hand otherwise.
export E2B_API_KEY=your_e2b_api_key
export WANDB_API_KEY=your_wandb_api_key   # optional; falls back to console logger

# 2. Prepare the dataset. Lands in
#    examples/train_integrations/harbor_pgc/data/Nemotron-Terminal-Synthetic-Tasks/.
uv run examples/train_integrations/harbor_pgc/prepare_harbor_dataset.py \
    --dataset nvidia/Nemotron-Terminal-Synthetic-Tasks

# 3. Build the 11 per-skill ghcr.io base images (one-time, ~30–60 min total).
#    Requires docker + a GitHub PAT with `write:packages` (via `gh auth`).
#    After push, flip each package to public at
#    https://github.com/users/<owner>?tab=packages.
bash examples/train_integrations/harbor_pgc/scripts/build_skill_images.sh

# 4. Rewrite every task.toml's docker_image to point at our ghcr images.
uv run -m examples.train_integrations.harbor_pgc.scripts.rewrite_task_dockerimage \
    examples/train_integrations/harbor_pgc/data/Nemotron-Terminal-Synthetic-Tasks

# 5. Sanity generation (no training, ~3–20 min depending on prompt difficulty).
bash examples/train_integrations/harbor_pgc/run_harbor_gen.sh

# 6. Smoke training (~1 hour, validates FSDP + weight sync end-to-end).
bash examples/train_integrations/harbor_pgc/run_nemotron_terminal_smoke.sh

# 7. Full training.
bash examples/train_integrations/harbor_pgc/run_nemotron_terminal.sh
# or fully-async:
bash examples/train_integrations/harbor_pgc/run_nemotron_terminal_fully_async.sh
```

### Dataset variants

`nvidia/Nemotron-Terminal-Synthetic-Tasks` ships three independent
shards under `skill_based/`. Same task format throughout
(`task.toml + instruction.md + environment/Dockerfile + tests/`),
same per-skill Dockerfile, so our 11 ghcr images cover all of them.

| Shard | Tasks | Skills | Notes |
|---|---|---|---|
| `mixed/*.tar.gz` | **5,984** | 6 (data_processing, data_science, debugging, file_operations, scientific_computing, security) | **Default — what `run_nemotron_terminal*.sh` point at.** Matches the old NeMo-RL recipe. |
| `easy.tar.gz` | 44,969 | 9 (+ data_querying, software_engineering, dependency_management) | Wider skill mix at moderate scale. Good for skill-ablation. |
| `medium_shard1.tar.gz` + `medium_shard2.tar.gz` | 203,749 | 11 (+ model_training, system_administration) | Full bulk synth pool, two difficulty tiers. Use only for large-scale scaling experiments. |

Bigger sister repo `nvidia/Nemotron-Terminal-Corpus` packages the same
synthetic_tasks/ in parquet form plus `dataset_adapters/code.parquet`,
`math.parquet`, and `swe.parquet` (real SWE-bench tasks bridged into
the harbor format). Worth a look when you want a SWE-bench eval number
to land in the paper.

Other harbor-format public datasets:
- `harborframework/terminal-bench-2.0` — 89 hand-written tasks; official tbench eval.
- `zai-org/terminal-bench-2-verified` — same 89 with instruction/env fixes.
- `laude-institute/sandboxes-tasks` — 94 hand-written from harbor's authors.

### Box-specific notes (dev box)

- `.python-version` pins CPython 3.12 — SkyRL's pinned wheel URLs use
  `python_version == '3.12'` markers; the system default 3.13 falls back to
  source builds that then trip torch's nvcc version check.
- We pass `CUDA_VISIBLE_DEVICES=0,1,2,3` to keep clear of GPU 7 which
  another user's stale `VLLM::EngineCore` is often sitting on.
- B300 GPUs (compute capability 10.3) handle Qwen3.5-9B at 256K context
  comfortably (linear-attention layers keep KV small).
