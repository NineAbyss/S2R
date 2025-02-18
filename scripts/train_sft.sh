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



REPO_DIR=./code

TOTAL_BATCH_SIZE=32
MICRO_TRAIN_BATCH_SIZE=1
MICRO_EVAL_BATCH_SIZE=1
NUM_GPUS=$NODE_NUM
SAVE_STEPS=50
EVAL_STEPS=2000


GRADIENT_CHECKPOINTING=True
BF16=True
LEARNING_RATE=5e-6
MODEL_MAX_LENGTH=8000
GRADIENT_ACCUMULATION_STEPS=$[$TOTAL_BATCH_SIZE/$MICRO_TRAIN_BATCH_SIZE/$NUM_GPUS]
MODEL_NAME_OR_PATH=Qwen/Qwen2.5-Math-7B
BASE_DIR=./saved_models
# distributed setting
MASTER_ADDR=$CHIEF_IP
MASTER_PORT=6000
declare -a DATA_PATH_LIST=(
"data/train_data/sft_qwen2.5_math_7B.json"
)
for DATA_PATH in "${DATA_PATH_LIST[@]}"; do
    echo "Processing dataset: ${DATA_PATH}"
    
    EXPERIMENT_NAME=$(basename ${MODEL_NAME_OR_PATH})_sft_$(basename ${DATA_PATH%.*})_$(date '+%Y%m%d_%H%M%S')
    OUTPUT_DIR=${BASE_DIR}/${EXPERIMENT_NAME}
    mkdir -p ${OUTPUT_DIR}
    
    TMP_DIR=${OUTPUT_DIR}/tmp
    mkdir -p $TMP_DIR

    echo $NODE_IP_LIST > ${TMP_DIR}/env.txt

    # -------------------------------------------------------------------------------------------
    deepspeed --hostfile ${TMP_DIR}/hostfile --master_addr ${MASTER_ADDR} --master_port=${MASTER_PORT} ${REPO_DIR}/src/sft/src/sft_weighted_with_kl.py \
        --output_dir ${OUTPUT_DIR} \
        --do_train True \
        --data_paths ${DATA_PATH} \
        --model_type qwen \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --model_max_length ${MODEL_MAX_LENGTH} \
        --remove_unused_columns False \
        --report_to tensorboard \
        --overwrite_output_dir True \
        --per_device_train_batch_size ${MICRO_TRAIN_BATCH_SIZE} \
        --per_device_eval_batch_size ${MICRO_EVAL_BATCH_SIZE} \
        --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
        --num_train_epochs 5 \
        --logging_strategy steps \
        --logging_steps 1 \
        --save_strategy epoch \
        --save_steps ${SAVE_STEPS} \
        --learning_rate ${LEARNING_RATE} \
        --evaluation_strategy no \
        --eval_steps ${EVAL_STEPS} \
        --warmup_steps 5 \
        --gradient_checkpointing ${GRADIENT_CHECKPOINTING} \
        --bf16 ${BF16} \
        --lm_kl_coeff 0.1 \
        --pad_labels_with_ignore \
        --deepspeed ${REPO_DIR}/configs/ds_stage3.json
done


