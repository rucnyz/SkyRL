set -ex

# PGC variant: e2b sandbox (cloud), credentials in research/pgc_swe/.env on
# our shared dev box. Override DEFAULT_PGC_ENV_FILE if your env lives elsewhere.
DEFAULT_PGC_ENV_FILE="${PGC_ENV_FILE:-/scratch/yuzhou/projects/RL/research/pgc_swe/.env}"
if [ -f "$DEFAULT_PGC_ENV_FILE" ]; then
  set -a; source "$DEFAULT_PGC_ENV_FILE"; set +a
fi
: "${E2B_API_KEY:?E2B_API_KEY must be set (in $DEFAULT_PGC_ENV_FILE or shell)}"
# Optional: WANDB_API_KEY also gets sourced; logger=console below if unset

# Override the stale CUDA_HOME from research/pgc_swe/.env (which points at a
# nonexistent /usr/local/cuda-12.9). Box has CUDA 13.2 toolkit; upstream's
# pinned wheel URLs (erictang000 forks, torch 2.11, cp312) ship prebuilt so
# no source build (and no nvcc check) at install time.
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# Pin GPUs (default 4,5,6,7 — GPU 0 is sometimes taken by panmz/prelude on this box)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

# Prepare dataset first (downloads from HuggingFace and extracts tasks):
# uv run examples/train_integrations/harbor_pgc/prepare_harbor_dataset.py --dataset open-thoughts/CodeContests

DATA_DIR="$HOME/data/harbor"
TRAIN_DATA="['$DATA_DIR/CodeContests']"

CHAT_TEMPLATE_PATH="$(dirname "$0")/../../../skyrl/train/utils/templates/qwen3_acc_thinking.jinja2"
TRIALS_DIR="$HOME/trials_run"

#----------------
# Infrastructure setup
#----------------
NUM_GPUS=4
MAX_MODEL_LEN=262144   # Qwen3.5-9B native max_position_embeddings (256K, no rope scaling).
                       # B300 has 288 GB; mostly-linear-attn means small KV footprint.
ENABLE_RATE_LIMITING=true  # Enable rate/concurrency limiting for trajectory submissions
TRAJECTORIES_PER_SECOND=5  # Maximum trajectories per second (must be >= 1.0, fractional values like 1.5 are supported). null or omit to disable rate limiting
MAX_CONCURRENCY=512        # Maximum concurrent trial.run() calls allowed (must be >= 1). null or omit to disable concurrency limiting

# Qwen3.5 = multimodal architecture (Qwen3_5ForConditionalGeneration). For
# text-only RL we set language_model_only=true on all three workers so they
# instantiate the text submodel. GDN/linear-attention layers + sample packing
# have a known bug (HF transformers#44910, QwenLM/Qwen3.5#104), so we disable
# sample packing too.
MODEL_NAME="Qwen/Qwen3.5-9B"
SERVED_NAME="Qwen3.5-9B"

# `--with "harbor[e2b]"` adds the e2b SDK on top of the project's harbor[daytona,modal]
# extras. Without this, harbor.environments.e2b raises ImportError on `from e2b import ...`.
#
# Note: we run on the new inference path (default _SKYRL_USE_NEW_INFERENCE=1).
# Our HarborGenerator reads the runtime ``proxy_url`` off the RemoteInferenceClient
# instead of the static config port, so the foreign-squatter-on-8000 problem
# that bit the legacy path no longer applies. See harbor_generator.py:__init__.
uv run --isolated --extra fsdp --extra harbor --with "harbor[e2b]" -m examples.train_integrations.harbor_pgc.entrypoints.main_harbor_generate \
  data.train_data=$TRAIN_DATA \
  data.val_data=$TRAIN_DATA \
  harbor_trial_config.trials_dir=$TRIALS_DIR \
  trainer.policy.model.path="$MODEL_NAME" \
  trainer.policy.language_model_only=true \
  trainer.ref.language_model_only=true \
  trainer.use_sample_packing=false \
  generator.inference_engine.served_model_name="$SERVED_NAME" \
  generator.inference_engine.language_model_only=true \
  generator.inference_engine.num_engines=$NUM_GPUS \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.enable_http_endpoint=true \
  generator.inference_engine.http_endpoint_host="127.0.0.1" \
  generator.inference_engine.http_endpoint_port=8000 \
  generator.sampling_params.max_generate_length=16384 \
  trainer.algorithm.max_seq_len=$MAX_MODEL_LEN \
  generator.inference_engine.engine_init_kwargs.max_model_len=$MAX_MODEL_LEN \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  generator.inference_engine.engine_init_kwargs.chat_template=$CHAT_TEMPLATE_PATH \
  trainer.algorithm.advantage_estimator="grpo" \
  generator.step_wise_trajectories=true \
  trainer.placement.colocate_all=false \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  trainer.train_batch_size=$NUM_GPUS \
  trainer.policy_mini_batch_size=$NUM_GPUS \
  trainer.logger=console \
  generator.rate_limit.enabled=$ENABLE_RATE_LIMITING \
  generator.rate_limit.trajectories_per_second=$TRAJECTORIES_PER_SECOND \
  generator.rate_limit.max_concurrency=$MAX_CONCURRENCY \
  $@
