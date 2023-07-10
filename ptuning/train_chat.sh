PRE_SEQ_LEN=128
LR=1e-3
NUM_GPUS=1

torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS ptuning/main.py \
    --do_train \
    --train_file ./data/record.json \
    --validation_file ./data/record.json \
    --preprocessing_num_workers 8 \
    --prompt_column prompt \
    --response_column response \
    --history_column history \
    --overwrite_cache \
    --model_name_or_path THUDM/chatglm2-6b \
    --output_dir ./test \
    --overwrite_output_dir \
    --max_source_length 1024 \
    --max_target_length 128 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --num_train_epochs 5 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \

