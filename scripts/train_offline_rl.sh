#! /bin/bash

# connection config
#---------------------------------------------------------------------------------
NET_TYPE="high"
export NCCL_IB_TIMEOUT=24
if [[ "${NET_TYPE}" = "low" ]]; then
    export NCCL_SOCKET_IFNAME=eth1
    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_HCA=mlx5_2:1,mlx5_2:1
    export NCCL_IB_SL=3
    export NCCL_CHECK_DISABLE=1
    export NCCL_P2P_DISABLE=0
    export NCCL_LL_THRESHOLD=16384
    export NCCL_IB_CUDA_SUPPORT=1
    export NCCL_DEBUG=INFO
else
    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_SL=3
    export NCCL_CHECK_DISABLE=1
    export NCCL_P2P_DISABLE=0
    export NCCL_IB_DISABLE=0
    export NCCL_LL_THRESHOLD=16384
    export NCCL_IB_CUDA_SUPPORT=1
    export NCCL_SOCKET_IFNAME=bond1
    export UCX_NET_DEVICES=bond1
    export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
    export NCCL_COLLNET_ENABLE=0
    export SHARP_COLL_ENABLE_SAT=0
    export NCCL_NET_GDR_LEVEL=2
    export NCCL_IB_QPS_PER_CONNECTION=4
    export NCCL_IB_TC=160
    export NCCL_PXN_DISABLE=1
fi


WORKSPACE=S2R/code

# model path
MODEL_DIR=xxxxx

MODEL_PATH=${MODEL_DIR}/xxxxx

REF_MODEL_PATH=${MODEL_DIR}/xxxxx


MODEL_OUTPUT_DIR=xxxxx/your_model_output_path

# datset path
DATA_DIR=./xxxxxx
TRAIN_DATA_NAME=your_train_data_name

TRAIN_DATA_PATH=${DATA_DIR}/${TRAIN_DATA_NAME}.json

# distributed setting
MASTER_ADDR=$CHIEF_IP
MASTER_PORT=6000

# generate evvironment
BATCH_SIZE=64
MICRO_BATCH_SIZE=1
NUM_GPUS=$NODE_NUM
GRADIENT_ACCUMULATION_STEP=$((BATCH_SIZE / NUM_GPUS / MICRO_BATCH_SIZE))

MODEL_MAX_LENGTH=8192

PADDING_SIDE="right"
TRUNCATION_SIDE="left"
POOLING_TYPE="last"
MODEL_TYPE=qwen
FORMAT_MODE=qwen_token


EPOCH=1
LEARNING_RATE=1e-6
WARMUP_STEPS=5
EVAL_STEPS=100
SAVE_STEPS=20

# loss params
USE_SFT_LOSS=false
LM_KL_COEFF=0.15
LM_SFT_COEFF=0.15

GRADIENT_CHECKPOINTING=true
BF16=true
FLASH_ATTN=false

# scale params 
USE_PROCESS_RL=false
USE_BONUS=true
USE_VERI_BONUS=true
REWARD_DELAY_FACTOR=0.6

# baseline method
USE_PREFIX_BASELINE=false
USE_LEVEL_BASELINE=false
USE_POSITION_BASELINE=false
USE_ACCURACY_BASELINE=true
USE_SOFTMAX_NORM=false
MIN_GROUP_BASELINE=10

DEBUG=false

# generate hostfile and pssh.hosts
TMP_DIR=${WORKSPACE}/tmp
mkdir -p $TMP_DIR
echo $NODE_IP_LIST > ${TMP_DIR}/env.txt
sed "s/:/ slots=/g" ${TMP_DIR}/env.txt | sed "s/,/\n/g" >  ${TMP_DIR}/hostfile
sed "s/:.//g" ${TMP_DIR}/env.txt | sed "s/,/\n/g" >  ${TMP_DIR}/pssh.hosts
DEEPSPEED=${WORKSPACE}/scripts/ds_config_A100_bf16.json


# output config
baseline_method="none"

if [ "$USE_PREFIX_BASELINE" = true ] && [ "$USE_ACCURACY_BASELINE" = true ]; then
    baseline_method="prefix_acc"
elif [ "$USE_PREFIX_BASELINE" = true ]; then
    baseline_method="prefix"
elif [ "$USE_LEVEL_BASELINE" = true ]; then
    baseline_method="level"
elif [ "$USE_POSITION_BASELINE" = true ]; then
    baseline_method="position"
elif [ "$USE_ACCURACY_BASELINE" = true ]; then
    baseline_method="accuracy"
fi

EXPERIMENT_NAME="offline_use_process${USE_PROCESS_RL}_${TRAIN_DATA_NAME}_epoch${EPOCH}_lr${LEARNING_RATE}_maxlen${MODEL_MAX_LENGTH}_batchsize${BATCH_SIZE}_klcoeff${LM_KL_COEFF}_usebonus${USE_BONUS}_baseline${baseline_method}_usesoftmax${USE_SOFTMAX_NORM}_mingroup${MIN_GROUP_BASELINE}$(date +'%m%d')"

OUTPUT_DIR=${MODEL_OUTPUT_DIR}/${EXPERIMENT_NAME}
LOGS_PATH=${OUTPUT_DIR}/logs

mkdir -p $OUTPUT_DIR
mkdir -p $LOGS_PATH

echo "begin experiment ${EXPERIMENT_NAME}"

# ----------------------------------------------------------------------
export CMD="deepspeed --hostfile ${TMP_DIR}/hostfile --master_addr ${MASTER_ADDR} --master_port=${MASTER_PORT}  ${WORKSPACE}/train_offline.py \
    --model_path ${MODEL_PATH} \
    --ref_model_path ${REF_MODEL_PATH} \
    --train_data_path ${TRAIN_DATA_PATH} \
    --label_names score tokens \
    --remove_unused_columns false \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs ${EPOCH} \
    --per_device_train_batch_size ${MICRO_BATCH_SIZE} \
    --per_device_eval_batch_size ${MICRO_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEP} \
    --evaluation_strategy no \
    --padding_side ${PADDING_SIDE} \
    --truncation_side ${TRUNCATION_SIDE} \
    --format_mode ${FORMAT_MODE} \
    --pooling_type ${POOLING_TYPE} \
    --max_length ${MODEL_MAX_LENGTH} \
    --save_strategy steps \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit 30 \
    --learning_rate ${LEARNING_RATE} \
    --warmup_steps ${WARMUP_STEPS} \
    --restart_step 0 \
    --step_num_per_stage 30 \
    --logging_steps 1 \
    --weight_decay 0. \
    --deepspeed ${DEEPSPEED} \
    --bf16 true \
    --tf32 true \
    --dump_data_path ${MODEL_OUTPUT_DIR}/${EXPERIMENT_NAME}/logs/train_data_log_$(date +'%H').jsonl \
    --gradient_checkpointing ${GRADIENT_CHECKPOINTING} \
    --debug_mode ${DEBUG} \
    --lm_kl_coeff ${LM_KL_COEFF} \
    --lm_sft_coeff ${LM_SFT_COEFF} \
    --reward_delay_factor ${REWARD_DELAY_FACTOR} \
    --clip_range 0.2 \
    --use_process_rl ${USE_PROCESS_RL} \
    --use_sft_loss ${USE_SFT_LOSS} \
    --use_bonus ${USE_BONUS} \
    --use_veri_bonus ${USE_VERI_BONUS} \
    --use_prefix_baseline ${USE_PREFIX_BASELINE} \
    --use_level_baseline ${USE_LEVEL_BASELINE} \
    --use_position_baseline ${USE_POSITION_BASELINE} \
    --use_accuracy_baseline ${USE_ACCURACY_BASELINE} \
    --use_softmax_norm ${USE_SOFTMAX_NORM} \
    --min_group_baseline ${MIN_GROUP_BASELINE}"


CURRENT_TIME=$(date +'%m-%d_%T')

echo $CMD
eval ${CMD} 2>&1 | tee -a ${LOGS_PATH}/log_${CURRENT_TIME}.txt
set +x
