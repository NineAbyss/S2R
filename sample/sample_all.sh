export CUDA_VISIBLE_DEVICES="0,1,2,3"

cd ./sample
PROMPT_TYPE="qwen25-math-cot"
MODEL_NAME_OR_PATH=your_model_path
OUTPUT_DIR=your_offline_samples_output_path

start=0 # start id
end=-1 # end id

OUT_FILE="${OUTPUT_DIR}/qwen25_math_7b_sft_rl1_start${start}_$(date +%Y%m%d).jsonl"
DATA_FILE="data/train_data/rl_data_offline.jsonl"

bash sh/sampling.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $OUTPUT_DIR $DATA_FILE $OUT_FILE $start $end float16