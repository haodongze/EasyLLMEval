#!/bin/bash

MODEL_PATH="/mnt/data/group/haodongze/internvl-verl/verl_internvl_work_dirs/internvl3_5_4b_custom/deepseek-r1-1_5b_multi_nodes_grpo_on_policy_distill_plus/global_step_78/actor/huggingface"
# 定义所有要跑的 data_name
DATA_NAMES=("aime2024" "aime2025-1" "aime2025-2")

# 遍历
for data_name in "${DATA_NAMES[@]}"; do
    echo "Running with data_name=${data_name}..."
    
    CUDA_VISIBLE_DEVICES='0,1,2,3' \
    python eval.py \
        --model_name_or_path ${MODEL_PATH} \
        --data_name "${data_name}" \
        --prompt_type "qwen-instruct" \
        --temperature 0.6 \
        --start_idx 0 \
        --end_idx -1 \
        --n_sampling 64 \
        --k 1 \
        --split "test" \
        --max_tokens 32768 \
        --seed 0 \
        --top_p 0.95 \
        --surround_with_messages
    
    echo "Finished ${data_name}"
done

echo "All done!"

DATA_NAMES=("amc" "math" "minerva" "olympiadbench")

# 遍历
for data_name in "${DATA_NAMES[@]}"; do
    echo "Running with data_name=${data_name}..."
    
    CUDA_VISIBLE_DEVICES='0,1,2,3' \
    python eval.py \
        --model_name_or_path ${MODEL_PATH} \
        --data_name "${data_name}" \
        --prompt_type "qwen-instruct" \
        --temperature 0.6 \
        --start_idx 0 \
        --end_idx -1 \
        --n_sampling 4 \
        --k 1 \
        --split "test" \
        --max_tokens 32768 \
        --seed 0 \
        --top_p 0.95 \
        --surround_with_messages
    
    echo "Finished ${data_name}"
done

echo "All done!"

