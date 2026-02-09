CUDA_VISIBLE_DEVICES='0,1,2,3' \
python eval.py \
--model_name_or_path "/mnt/data/group/models/DeepSeek-R1-Distill-Qwen-1.5B" \
--data_name "aime" \
--prompt_type "qwen-instruct" \
--temperature 0.6 \
--start_idx 0 \
--end_idx -1 \
--n_sampling 1 \
--k 1 \
--split "test" \
--max_tokens 32768 \
--seed 0 \
--top_p 0.95 \
--surround_with_messages \

