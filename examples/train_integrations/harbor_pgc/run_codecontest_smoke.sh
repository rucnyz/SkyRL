set -ex

# Smoke test for the harbor_pgc training loop. Same wiring as
# run_codecontest.sh but minimal knobs — just enough to validate:
#   1. FSDP shard of Qwen3.5-9B onto 4×B300 without OOM
#   2. One GRPO step computes a loss and backwards
#   3. NCCL weight sync from trainer back to vllm-router engines succeeds
#   4. Step-wise rollout_details flow end-to-end
#
# We intentionally:
#   - shrink MAX_MODEL_LEN to 32K (we're not testing long-context here)
#   - shrink MINI_BATCH_SIZE / N_SAMPLES_PER_PROMPT for fast iteration
#   - disable eval, ckpt, wandb
#   - log to console
# Run for a couple of GRPO steps then ctrl-c (no `max_steps` knob in GRPO).

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
# Dataset
#-----------------------
DATA_DIR="$HOME/data/harbor"
# 16-task symlinked subset of CodeContests so 1 epoch ≈ a handful of steps.
# Created with: mkdir -p CodeContests_smoke16 && ls CodeContests | head -16 | xargs -I{} ln -s "$PWD/CodeContests/{}" CodeContests_smoke16/
TRAIN_DATA="['$DATA_DIR/CodeContests_smoke16']"
EVAL_DATA="['$DATA_DIR/CodeContests_smoke16']"  # eval is disabled below; pointing at train avoids needing a second dataset

#-----------------------
# Storage
#-----------------------
RUN_NAME="codecontest-smoke"
STORAGE_ROOT="$HOME/skyrl_runs/$RUN_NAME"
TRIALS_DIR="$STORAGE_ROOT/trials_run"
CKPTS_DIR="$STORAGE_ROOT/ckpts"
EXPORTS_DIR="$STORAGE_ROOT/exports"
LOG_DIR="$STORAGE_ROOT/logs"

#-----------------------
# Model + training (smoke values, not real-experiment)
#-----------------------
MODEL_NAME="Qwen/Qwen3.5-9B"
SERVED_NAME="Qwen3.5-9B"
MAX_MODEL_LEN=32768
N_SAMPLES_PER_PROMPT=2
MINI_BATCH_SIZE=4

LOSS_REDUCTION="token_mean"
GRPO_NORM_BY_STD=false
USE_KL_LOSS=false
APPLY_OVERLONG_FILTERING=true
CHAT_TEMPLATE_PATH="$(dirname "$0")/../../../skyrl/train/utils/templates/qwen3_acc_thinking.jinja2"
TIS_TYPE=token
TIS_IMP_RATIO_CAP=2.0

#----------------
# Infra
#----------------
NUM_POLICY_GPUS=4
NUM_INFERENCE_ENGINES=4
TP_SIZE=1
ENABLE_RATE_LIMITING=true
TRAJECTORIES_PER_SECOND=5
MAX_CONCURRENCY=64   # smoke: keep e2b pressure low

uv run --isolated --extra fsdp --extra harbor --with "harbor[e2b]" -m examples.train_integrations.harbor_pgc.entrypoints.main_harbor \
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
  trainer.eval_before_train=false \
  trainer.eval_interval=999999 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$MINI_BATCH_SIZE \
  trainer.policy_mini_batch_size=$MINI_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=999999 \
  trainer.hf_save_interval=999999 \
  trainer.algorithm.max_seq_len=$MAX_MODEL_LEN \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  generator.step_wise_trajectories=true \
  generator.merge_stepwise_output=true \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  generator.eval_n_samples_per_prompt=1 \
  generator.apply_overlong_filtering=$APPLY_OVERLONG_FILTERING \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  trainer.logger=console \
  trainer.project_name=harbor_pgc_smoke \
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
  generator.sampling_params.max_generate_length=4096 \
  generator.rate_limit.enabled=$ENABLE_RATE_LIMITING \
  generator.rate_limit.trajectories_per_second=$TRAJECTORIES_PER_SECOND \
  generator.rate_limit.max_concurrency=$MAX_CONCURRENCY \
  "$@"
