## Harbor Integration — PGC variant

Fork of `examples/train_integrations/harbor/` used by the PGC paper experiments.
RL training with [Harbor](https://github.com/laude-institute/harbor) as the
environment + reward source, agent rollouts running inside an
[E2B](https://e2b.dev/) cloud sandbox, and a `Qwen/Qwen3.5-9B` policy.

### How this fork differs from upstream

| | upstream `harbor/` | this `harbor_pgc/` |
|---|---|---|
| Sandbox backend | daytona / modal | **e2b** |
| Model | Qwen3-8B (8K ctx) | Qwen3.5-9B (256K ctx) |
| Inference path | SkyRL legacy HTTP endpoint | **upstream-default new path** (vllm-router) |
| LLM client | LiteLLM → `/v1/chat/completions` | **`SkyRLNativeLLM` → `/skyrl/v1/generate`** |
| Agent class | stock `Terminus2` | `SkyRLTerminus2` via `agent.import_path` |

The new-inference / native-LLM swap is the load-bearing change. vllm-router's
OpenAI-compatible `/v1/chat/completions` route silently drops the vllm
extension fields (`prompt_token_ids`, `completion_token_ids`, `logprobs`)
that step-wise RL training needs. We instead talk to the same router's
`/skyrl/v1/generate` endpoint, which preserves them. Chat-template
application moves client-side — see `llms/skyrl_native_llm.py`.

### Structure

```
harbor_pgc/
  agents/skyrl_terminus_2.py       Terminus-2 subclass; overrides _init_llm
                                   to dispatch llm_backend="skyrl" to our
                                   SkyRLNativeLLM (loaded via harbor's
                                   official agent.import_path extension
                                   point — no monkey-patching).
  llms/skyrl_native_llm.py         BaseLLM subclass that applies the chat
                                   template with the model's HF tokenizer
                                   and POSTs token_ids to
                                   {proxy_url}/skyrl/v1/generate.
  harbor_generator.py              HarborGenerator: reads proxy_url off the
                                   RemoteInferenceClient at runtime, wires
                                   agent.import_path + agent.kwargs so
                                   harbor instantiates SkyRLTerminus2.
  dataset.py                       HarborTaskDataset: loads task directory
                                   paths from CodeContests-style dumps.
  prepare_harbor_dataset.py        Downloads + extracts datasets from HuggingFace.
  prebuild_e2b_templates.py        Pre-builds the per-task e2b template
                                   alias used by harbor's E2BEnvironment.
                                   Must be run once before training so we
                                   don't fight e2b's parallel-build rate
                                   limiter at runtime.
  harbor_trial_config/default.yaml Harbor TrialConfig template (e2b sandbox,
                                   Qwen3.5-9B model_info).
  entrypoints/
    main_harbor.py                 Full GRPO training entrypoint.
    main_harbor_fully_async.py     Fully-async training entrypoint.
    main_harbor_generate.py        Generation-only sanity entrypoint.
  run_codecontest.sh               Code-contest training (sync).
  run_codecontest_fully_async.sh   Code-contest training (fully async).
  run_harbor_gen.sh                Generation-only sanity launcher.
```

### Quick Start

```bash
cd SkyRL

# 1. Credentials. We default to sourcing PGC_ENV_FILE
#    (/scratch/yuzhou/projects/RL/research/pgc_swe/.env on this dev box) which
#    sets E2B_API_KEY and WANDB_API_KEY. Set them directly otherwise.
export E2B_API_KEY=your_e2b_api_key
export WANDB_API_KEY=your_wandb_api_key   # optional, fall back to console logger

# 2. Prepare dataset (skip if already on disk).
uv run examples/train_integrations/harbor_pgc/prepare_harbor_dataset.py \
    --dataset open-thoughts/CodeContests

# 3. Pre-build the e2b template aliases (one-time, ~1 min/task sequentially).
#    Without this, multiple concurrent trials race on the first sandbox.create
#    and e2b's rate limiter cancels most of them, leaving zombie aliases that
#    404 on subsequent runs.
uv run --isolated --extra harbor --with "harbor[e2b]" \
    -m examples.train_integrations.harbor_pgc.prebuild_e2b_templates \
    "$HOME/data/harbor/CodeContests" [--limit N]

# 4. Sanity generation (no training, ~3-20 min depending on prompt difficulty).
bash examples/train_integrations/harbor_pgc/run_harbor_gen.sh

# 5. Training.
bash examples/train_integrations/harbor_pgc/run_codecontest.sh
# or the fully-async variant:
bash examples/train_integrations/harbor_pgc/run_codecontest_fully_async.sh
```

### Box-specific notes (dev box)

- `.python-version` pins CPython 3.12 — SkyRL's pinned wheel URLs use
  `python_version == '3.12'` markers; the system default 3.13 falls back to
  source builds that then trip torch's nvcc version check.
- We pass `CUDA_VISIBLE_DEVICES=0,1,2,3` to keep clear of GPU 7 which
  another user's stale `VLLM::EngineCore` is often sitting on.
- B300 GPUs (compute capability 10.3) handle Qwen3.5-9B at 256K context
  comfortably (linear-attention layers keep KV small).
