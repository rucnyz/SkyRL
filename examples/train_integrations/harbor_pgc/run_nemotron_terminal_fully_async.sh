set -ex

# PGC variant: e2b sandbox + Qwen3.5-9B + 256K context. Source the shared
# PGC env file (E2B_API_KEY, WANDB_API_KEY) — same convention as
# run_harbor_gen.sh.
DEFAULT_PGC_ENV_FILE="${PGC_ENV_FILE:-/scratch/yuzhou/projects/RL/research/pgc_swe/.env}"
if [ -f "$DEFAULT_PGC_ENV_FILE" ]; then
  set -a; source "$DEFAULT_PGC_ENV_FILE"; set +a
fi
: "${E2B_API_KEY:?E2B_API_KEY must be set (in $DEFAULT_PGC_ENV_FILE or shell)}"

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

#-----------------------
# Dataset setup
#-----------------------
# Prepare dataset first (downloads from HuggingFace and extracts tasks into
# examples/train_integrations/harbor_pgc/data/<repo>/):
# uv run examples/train_integrations/harbor_pgc/prepare_harbor_dataset.py \
#     --dataset nvidia/Nemotron-Terminal-Synthetic-Tasks
DATA_DIR="$(dirname "$0")/data"
TRAIN_DATA="['$DATA_DIR/Nemotron-Terminal-Synthetic-Tasks']"
EVAL_DATA="['$DATA_DIR/Nemotron-Terminal-Synthetic-Tasks']"  # TODO: carve out a held-out val split

#-----------------------
# Directory setup
#-----------------------
RUN_NAME="codecontest-fullyasync"
STORAGE_ROOT="$HOME/skyrl_runs/$RUN_NAME"
TRIALS_DIR="$STORAGE_ROOT/trials_run"
CKPTS_DIR="$STORAGE_ROOT/ckpts"
EXPORTS_DIR="$STORAGE_ROOT/exports"
LOG_DIR="$STORAGE_ROOT/logs"

#-----------------------
# Model + training setup
#-----------------------
MODEL_NAME="Qwen/Qwen3.5-9B"
SERVED_NAME="Qwen3.5-9B"
MAX_MODEL_LEN=262144   # Qwen3.5-9B native max_position_embeddings (256K).

N_SAMPLES_PER_PROMPT=8
MINI_BATCH_SIZE=16

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

# -------------------------
# Fully-async knobs.
# All knobs are tuned for 1x8xH100 node for Qwen3-8B, please adjust accordingly.
# Constraint: mini_batch_size <= num_parallel_generation_workers <= mini_batch_size * (max_staleness_steps + 1)
# Can increase num_parallel_generation_workers based on your hardware resources (e.g. KV cache size).
# -------------------------
MAX_STALENESS_STEPS=4
NUM_PARALLEL_GENERATION_WORKERS=$(( MINI_BATCH_SIZE * 2 ))

#----------------
# Infrastructure setup. Tuned for 1n4g (the dev box default).
#----------------
NUM_INFERENCE_ENGINES=2
TP_SIZE=1
NUM_POLICY_GPUS=2
ENABLE_RATE_LIMITING=true
TRAJECTORIES_PER_SECOND=5
# E2B account hard-caps concurrent sandboxes at 100 (429 beyond). Cap well below.
MAX_CONCURRENCY=64

# Run SkyRL command — talks to vllm-router on the new inference path via
# SkyRLTerminus2 + SkyRLNativeLLM (see harbor_pgc/README.md).
uv run --isolated --extra fsdp --extra harbor --with "harbor[e2b]" -m examples.train_integrations.harbor_pgc.entrypoints.main_harbor_fully_async \
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
  trainer.fully_async.max_staleness_steps=$MAX_STALENESS_STEPS \
  trainer.fully_async.num_parallel_generation_workers=$NUM_PARALLEL_GENERATION_WORKERS \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.loss_reduction=$LOSS_REDUCTION \
  trainer.algorithm.grpo_norm_by_std=$GRPO_NORM_BY_STD \
  trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
  trainer.algorithm.off_policy_correction.tis_ratio_type=$TIS_TYPE \
  trainer.algorithm.off_policy_correction.token_tis_ratio_clip_high=$TIS_IMP_RATIO_CAP \
  trainer.placement.colocate_all=false \
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
  trainer.epochs=3 \
  trainer.eval_batch_size=128 \
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
  generator.inference_engine.gpu_memory_utilization=0.9 \
  trainer.logger=wandb \
  trainer.project_name=harbor \
  trainer.run_name=$RUN_NAME \
  trainer.resume_mode=latest \
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
