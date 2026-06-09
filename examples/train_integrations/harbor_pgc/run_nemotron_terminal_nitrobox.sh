set -ex

# PGC variant: nitrobox sandbox (local user-namespace sandboxes via the
# rucnyz/harbor fork) + Qwen3.5-9B + 256K context. No cloud sandbox account
# needed — tasks' environment/Dockerfile are built locally through nitrobox's
# embedded buildkit (per-skill base layers shared via layer cache).
#
# Smoke-scale defaults: 2 GPUs colocated, 8-prompt batch x 4 samples
# (32 trials/step) against the 16-task smoke dataset. Scale knobs up once the
# pipeline is verified.
#
# Requires: AppArmor profile for nitrobox-core on Ubuntu 24.04+
# (apparmor_restrict_unprivileged_userns=1) — see /etc/apparmor.d/nitrobox-core.
DEFAULT_PGC_ENV_FILE="${PGC_ENV_FILE:-/scratch/yuzhou/projects/RL/research/pgc_swe/.env}"
if [ -f "$DEFAULT_PGC_ENV_FILE" ]; then
  set -a; source "$DEFAULT_PGC_ENV_FILE"; set +a
fi

# Box has CUDA 13.2 toolkit only; upstream's prebuilt wheels mean no nvcc
# check fires at install. Pin CUDA_HOME so torch's cpp_extension picks the
# right toolchain at runtime.
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# Smoke run: 2 GPUs are enough for Qwen3.5-9B colocated (GPU 7 is often
# pinned by another user's stale vllm).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

#-----------------------
# Dataset setup
#-----------------------
# Prepare dataset first (downloads from HuggingFace and extracts tasks into
# examples/train_integrations/harbor_pgc/data/<repo>/):
# uv run examples/train_integrations/harbor_pgc/prepare_harbor_dataset.py \
#     --dataset nvidia/Nemotron-Terminal-Synthetic-Tasks
# Absolute path: relative paths resolve inside ray's packaged working_dir
# copy, where data/ is not shipped ("Path does not exist" -> dataset size 0).
DATA_DIR="$(cd "$(dirname "$0")" && pwd)/data"
TRAIN_DATA="['$DATA_DIR/Nemotron-Terminal-Synthetic-Tasks_smoke16']"
EVAL_DATA="['$DATA_DIR/Nemotron-Terminal-Synthetic-Tasks_smoke16']"

#-----------------------
# Directory setup
#-----------------------
RUN_NAME="${RUN_NAME:-nitrobox_smoke_v1}"
STORAGE_ROOT="$HOME/skyrl_runs/$RUN_NAME"
TRIALS_DIR="$STORAGE_ROOT/trials_run"
CKPTS_DIR="$STORAGE_ROOT/ckpts"
EXPORTS_DIR="$STORAGE_ROOT/exports"
LOG_DIR="$STORAGE_ROOT/logs"

#-----------------------
# Model + training setup
#-----------------------
# Qwen3.5-9B is multimodal (Qwen3_5ForConditionalGeneration); use the
# text-only submodel via language_model_only=true on all three workers, and
# disable sample packing (broken on GDN/linear-attention layers — see
# transformers#44910, QwenLM/Qwen3.5#104).
MODEL_NAME="Qwen/Qwen3.5-9B"
SERVED_NAME="Qwen3.5-9B"
MAX_MODEL_LEN=262144   # Qwen3.5-9B native max_position_embeddings (256K).

N_SAMPLES_PER_PROMPT=4
MINI_BATCH_SIZE=8

# Algorithmic parameters
LOSS_REDUCTION="token_mean"  # with step-wise training, we have to use token_mean to be prefix-merge-invariant
GRPO_NORM_BY_STD=false
USE_KL_LOSS=false
APPLY_OVERLONG_FILTERING=true

# Essentially achieves interleaved thinking (does not strip thinking tokens). Allows our step-wise
# training to be able to merge more step-wise outputs and hence speed up training.
# If you change the model you train, please change it accordingly, and decide if you need to make
# modifications.
CHAT_TEMPLATE_PATH="$(dirname "$0")/../../../skyrl/train/utils/templates/qwen3_acc_thinking.jinja2"

# TIS corrections
TIS_TYPE=token
TIS_IMP_RATIO_CAP=2.0

#----------------
# Infrastructure setup
#----------------
NUM_POLICY_GPUS=2
NUM_INFERENCE_ENGINES=2
TP_SIZE=1
ENABLE_RATE_LIMITING=true  # Enable rate/concurrency limiting for trajectory submissions
TRAJECTORIES_PER_SECOND=2  # Maximum trajectories per second (must be >= 1.0, fractional values like 1.5 are supported).
# nitrobox sandboxes are local processes (box: 256 cores / 3 TB RAM); the cap
# here just bounds concurrent image builds + agent loops.
MAX_CONCURRENCY=16

# Run SkyRL command — talks to vllm-router on the new inference path via
# SkyRLTerminus2 + SkyRLNativeLLM (see harbor_pgc/README.md).
uv run --isolated --extra fsdp --extra harbor -m examples.train_integrations.harbor_pgc.entrypoints.main_harbor \
  data.train_data=$TRAIN_DATA \
  data.val_data=$EVAL_DATA \
  trainer.policy.model.path="$MODEL_NAME" \
  trainer.policy.language_model_only=true \
  trainer.ref.language_model_only=true \
  trainer.use_sample_packing=false \
  generator.inference_engine.served_model_name="$SERVED_NAME" \
  generator.inference_engine.language_model_only=true \
  harbor_trial_config.trials_dir=$TRIALS_DIR \
  trainer.export_path=$EXPORTS_DIR \
  trainer.ckpt_path=$CKPTS_DIR \
  trainer.log_path=$LOG_DIR \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.loss_reduction=$LOSS_REDUCTION \
  trainer.algorithm.grpo_norm_by_std=$GRPO_NORM_BY_STD \
  trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
  trainer.algorithm.off_policy_correction.tis_ratio_type=$TIS_TYPE \
  trainer.algorithm.off_policy_correction.token_tis_ratio_clip_high=$TIS_IMP_RATIO_CAP \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp \
  trainer.placement.policy_num_nodes=1 \
  trainer.placement.ref_num_nodes=1 \
  trainer.placement.policy_num_gpus_per_node=$NUM_POLICY_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_POLICY_GPUS \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$TP_SIZE \
  generator.inference_engine.engine_init_kwargs.chat_template=$CHAT_TEMPLATE_PATH \
  generator.inference_engine.engine_init_kwargs.max_model_len=$MAX_MODEL_LEN \
  generator.inference_engine.engine_init_kwargs.enable_log_requests=false \
  trainer.epochs=1 \
  trainer.eval_batch_size=16 \
  trainer.eval_before_train=false \
  trainer.eval_interval=100 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$MINI_BATCH_SIZE \
  trainer.policy_mini_batch_size=$MINI_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=5 \
  trainer.max_ckpts_to_keep=5 \
  trainer.hf_save_interval=5 \
  trainer.algorithm.max_seq_len=$MAX_MODEL_LEN \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  generator.step_wise_trajectories=true \
  generator.sampling_params.max_generate_length=16384 \
  generator.merge_stepwise_output=true \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  generator.eval_n_samples_per_prompt=2 \
  generator.apply_overlong_filtering=$APPLY_OVERLONG_FILTERING \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  trainer.logger=wandb \
  trainer.project_name=harbor \
  trainer.run_name=$RUN_NAME \
  trainer.resume_mode=none \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.batched=false \
  generator.inference_engine.enforce_eager=false \
  generator.inference_engine.enable_http_endpoint=true \
  generator.inference_engine.http_endpoint_host=127.0.0.1 \
  generator.inference_engine.http_endpoint_port=8000 \
  generator.rate_limit.enabled=$ENABLE_RATE_LIMITING \
  generator.rate_limit.trajectories_per_second=$TRAJECTORIES_PER_SECOND \
  generator.rate_limit.max_concurrency=$MAX_CONCURRENCY \
  "$@"
