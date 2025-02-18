export CUDA_VISIBLE_DEVICES="0,1,2,3"
cd tools/qwen_eval/eval
PROMPT_TYPE="qwen25-math-cot"
MODEL_DIR=$1  


for MODEL_NAME_OR_PATH in "$MODEL_DIR"/*/ ; do
    if [ -d "$MODEL_NAME_OR_PATH" ]; then  
        LAST_TWO_DIRS="$(basename $(dirname ${MODEL_NAME_OR_PATH}))-$(basename ${MODEL_NAME_OR_PATH})"
        OUTPUT_DIR="tools/qwen_eval/eval/evaluation_results/${LAST_TWO_DIRS}"
        echo "Evaluating model: $MODEL_NAME_OR_PATH"
        bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $OUTPUT_DIR
    fi
done

