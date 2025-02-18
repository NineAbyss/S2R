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



DEBUG=false
# env config
#---------------------------------------------------------------------------------
MASTER_ADDR=$CHIEF_IP
MASTER_PORT=6000
NUM_GPUS=$NODE_NUM

# model input config
#---------------------------------------------------------------------------------
WORKSPACE=./code
export PYTHONPATH=${WORKSPACE}

MODEL_PATH=/PATH/TO/YOUR/MODEL/AFTER/SFT
REF_MODEL_PATH=/PATH/TO/YOUR/MODEL/AFTER/SFT

MODEL_OUTPUT_DIR=./saved_models
# data config
#---------------------------------------------------------------------------------


# training setups
#---------------------------------------------------------------------------------
BATCH_SIZE=64
MICRO_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEP=$((BATCH_SIZE / NUM_GPUS / MICRO_BATCH_SIZE))

MAX_LENGTH=8000

PADDING_SIDE="right"
TRUNCATION_SIDE="left"
POOLING_TYPE="last"

EPOCH=20
LEARNING_RATE=5e-7
WARMUP_STEPS=5
EVAL_STEPS=100
SAVE_STEPS=50

FLASH_ATTN=false

# generate environment, hostfile and pssh.hosts
TMP_DIR=${WORKSPACE}/tmp
mkdir -p $TMP_DIR
echo $NODE_IP_LIST > ${TMP_DIR}/env.txt
sed "s/:/ slots=/g" ${TMP_DIR}/env.txt | sed "s/,/\n/g" >  ${TMP_DIR}/hostfile
sed "s/:.//g" ${TMP_DIR}/env.txt | sed "s/,/\n/g" >  ${TMP_DIR}/pssh.hosts
DEEPSPEED=${WORKSPACE}/configs/ds_config_rl.json


EXPERIMENT_NAME=Qwen2.5_MATH$(date +'%m%d')

OUTPUT_DIR=${MODEL_OUTPUT_DIR}/${EXPERIMENT_NAME}
LOGS_PATH=${OUTPUT_DIR}/logs

mkdir -p $OUTPUT_DIR
mkdir -p $LOGS_PATH

echo "begin experiment ${EXPERIMENT_NAME}"

RL_DATA_PATH=./data/train_data/rl_data_qwen2.5.jsonl


export CMD="deepspeed --hostfile ${TMP_DIR}/hostfile --master_addr ${MASTER_ADDR} --master_port=${MASTER_PORT}  ${WORKSPACE}/train_trl.py \
    --model_path ${MODEL_PATH} \
    --ref_model_path ${REF_MODEL_PATH} \
    --label_names score tokens \
    --remove_unused_columns false \
    --output_dir ${OUTPUT_DIR} \
    --rl_data_path ${RL_DATA_PATH} \
    --num_train_epochs ${EPOCH} \
    --per_device_train_batch_size ${MICRO_BATCH_SIZE} \
    --per_device_eval_batch_size ${MICRO_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEP} \
    --evaluation_strategy no \
    --padding_side ${PADDING_SIDE} \
    --truncation_side ${TRUNCATION_SIDE} \
    --pooling_type ${POOLING_TYPE} \
    --max_length ${MAX_LENGTH} \
    --save_strategy steps \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit 30 \
    --learning_rate ${LEARNING_RATE} \
    --warmup_steps ${WARMUP_STEPS} \
    --restart_step 0 \
    --logging_steps 1 \
    --inner_step_num 1 \
    --weight_decay 0. \
    --deepspeed ${DEEPSPEED} \
    --bf16 true \
    --tf32 true \
    --dump_data_path ${MODEL_OUTPUT_DIR}/${EXPERIMENT_NAME}/logs/train_data_log_$(date +'%H').jsonl \
    --gradient_checkpointing True \
    --use_instance_level True \
    --debug_mode ${DEBUG} \
    --kl_coef 0.01 \
    --clip_range 0.2"

CURRENT_TIME=$(date +'%m-%d_%T')

echo $CMD
eval ${CMD} 2>&1 | tee -a ${LOGS_PATH}/log_${CURRENT_TIME}.txt
set +x
