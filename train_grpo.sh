#!/bin/bash
export CUDA_VISIBLE_DEVICES=4,5
export HF_ENDPOINT=https://hf-mirror.com
export DATA_DIR='/data1/wk/search-r1/StepSearch/data/rl'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export WAND_PROJECT='WK-search'
# wandb 断点继续
# export WANDB_RUN_ID='ebzm88mr'

# export BASE_MODEL='Qwen/Qwen2.5-3B'
# export BASE_MODEL='Qwen/Qwen2.5-3B'
# export EXPERIMENT_NAME=nq-mysearch-grpo-qwen2.5-3b-em
# export BASE_MODEL='Qwen/Qwen2.5-7B'
export BASE_MODEL='/data1/wk/search-r1/verl-main/checkpoints/WK-VERL/sft-qwen2.5-7b-by-verl/sft_2000_merge'
# export BASE_MODEL='/data1/wk/search-r1/StepSearch/verl_checkpoints/wk-search-grpo-qwen2.5-7b/actor/global_step_170'
export EXPERIMENT_NAME=wk-search-grpo-qwen2.5-7b

# set -x
export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues

########     关键参数配置start      ##########
export TRAIN_BATCH_SIZE=32
export VAL_BATCH_SIZE=50
export PPO_MINI_BATCH_SIZE=16
export TENSOR_MODEL_PARALLEL_SIZE=2 # 将一个大模型分成几份，同时运行在多张显卡上，必须是显卡数量的约数
export N_AGENT=4 # 每个问题，模型生成多少条路径
export N_GPUS_PER_NODE=2 # 使用的显卡数量
export SAVE_FREQ=10 # 保存频率
export TEST_FREQ=10 # 测试频率 100条数据，VAL_BATCH_SIZE=20 大概15分钟
export MAX_CKPT_TO_KEEP=4 # 最大checkpoint保存数量
export LR_WARMUP_STEPS_RATIO=0.285 # 学习率预热比例 = total_training_steps * LR_WARMUP_STEPS_RATIO
export VAL_BEFORE_TRAIN=true # 训练前验证
export SKIP_VALIDATION=false # 训练后验证
export PARAM_OFFLOAD=true
export LEARNING_RATE=5e-7
export GPU_MEMORY_UTILIZATION=0.6
########     关键参数配置end      ##########

# max_prompt_length = (config['training']['max_start_length'] + config['training']['max_response_length'] * (config['training']['max_turns'] - 1) + config['training']['max_obs_length'] * config['training']['max_turns'])

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=1500 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=$LR_WARMUP_STEPS_RATIO \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.fsdp_config.param_offload=$PARAM_OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.grad_offload=$PARAM_OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$PARAM_OFFLOAD \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_MODEL_PARALLEL_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.ref.log_prob_micro_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=$PARAM_OFFLOAD \
    actor_rollout_ref.actor.kl_loss_coef=0.0001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent=$N_AGENT \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger=['wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=$VAL_BEFORE_TRAIN \
    +trainer.skip_validation=$SKIP_VALIDATION \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1000 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    max_turns=5 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee $EXPERIMENT_NAME.log

# +actor_rollout_ref.model.lora_rank=64 \
# +actor_rollout_ref.model.lora_alpha=128 \
# +actor_rollout_ref.model.target_modules=all-linear \
# +actor_rollout_ref.model.use_shm=True \
# actor_rollout_ref.rollout.load_format=auto \
