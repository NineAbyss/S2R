set -ex
# export CUDA_VISIBLE_DEVICES=7
PROMPT_TYPE=$1
MODEL_NAME_OR_PATH=$2
OUTPUT_DIR=$3
DATA_FILE=$4
OUT_FILE=$5
start=$6
end=$7
dtype=$8

NUM_TEST_SAMPLE=-1

# English open datasets
# math500,gsm8k,olympiadbench,aime24,amc23,college_math
DATA_NAME="math500"
TOKENIZERS_PARALLELISM=false \

python3 -u math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --data_file ${DATA_FILE} \
    --output_dir ${OUTPUT_DIR} \
    --out_file ${OUT_FILE} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --dtype ${dtype} \
    --max_tokens_per_call 8192 \
    --seed 0 \
    --temperature 0.8 \
    --n_sampling 8 \
    --top_p 1 \
    --start ${start} \
    --end ${end} \
    --split_num 3 \
    --use_vllm \
    --save_outputs 

# # English multiple-choice datasets
# DATA_NAME="aqua,sat_math,mmlu_stem"
# TOKENIZERS_PARALLELISM=false \
# python3 -u math_eval.py \
#     --model_name_or_path ${MODEL_NAME_OR_PATH} \
#     --data_name ${DATA_NAME} \
#     --output_dir ${OUTPUT_DIR} \
#     --split ${SPLIT} \
#     --prompt_type ${PROMPT_TYPE} \
#     --num_test_sample ${NUM_TEST_SAMPLE} \
#     --seed 0 \
#     --temperature 0 \
#     --n_sampling 1 \
#     --top_p 1 \
#     --start 0 \
#     --end -1 \
#     --use_vllm \
#     --save_outputs \
#     --overwrite \
#     --num_shots 5

# # Chinese gaokao collections
# DATA_NAME="gaokao2024_I,gaokao2024_II,gaokao2024_mix,gaokao_math_cloze,gaokao_math_qa"
# TOKENIZERS_PARALLELISM=false \
# python3 -u math_eval.py \
#     --model_name_or_path ${MODEL_NAME_OR_PATH} \
#     --data_name ${DATA_NAME} \
#     --output_dir ${OUTPUT_DIR} \
#     --split ${SPLIT} \
#     --prompt_type ${PROMPT_TYPE} \
#     --num_test_sample ${NUM_TEST_SAMPLE} \
#     --seed 0 \
#     --temperature 0 \
#     --n_sampling 1 \
#     --top_p 1 \
#     --start 0 \
#     --end -1 \
#     --use_vllm \
#     --save_outputs \
#     --overwrite \
#     --adapt_few_shot

# # Chinese other datasets
# DATA_NAME="cmath,cn_middle_school"
# TOKENIZERS_PARALLELISM=false \
# python3 -u math_eval.py \
#     --model_name_or_path ${MODEL_NAME_OR_PATH} \
#     --data_name ${DATA_NAME} \
#     --output_dir ${OUTPUT_DIR} \
#     --split ${SPLIT} \
#     --prompt_type ${PROMPT_TYPE} \
#     --num_test_sample ${NUM_TEST_SAMPLE} \
#     --seed 0 \
#     --temperature 0 \
#     --n_sampling 1 \
#     --top_p 1 \
#     --start 0 \
#     --end -1 \
#     --use_vllm \
#     --save_outputs \
#     --overwrite \
#     --adapt_few_shot


# # English competition datasets
# DATA_NAME="aime24"
# TOKENIZERS_PARALLELISM=false \
# python3 -u math_eval.py \
#     --model_name_or_path ${MODEL_NAME_OR_PATH} \
#     --data_name ${DATA_NAME} \
#     --output_dir ${OUTPUT_DIR} \
#     --split ${SPLIT} \
#     --prompt_type ${PROMPT_TYPE} \
#     --num_test_sample ${NUM_TEST_SAMPLE} \
#     --seed 0 \
#     --temperature 0 \
#     --n_sampling 1 \
#     --top_p 1 \
#     --start 0 \
#     --end -1 \
#     --use_vllm \
#     --save_outputs 
