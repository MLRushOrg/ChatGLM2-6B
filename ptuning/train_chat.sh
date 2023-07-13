PRE_SEQ_LEN=128
LR=5e-4
NUM_GPUS=1

torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS ptuning/main.py \
    --do_train \
    --train_file ./data/record.json \
    --preprocessing_num_workers 8 \
    --prompt_column prompt \
    --response_column response \
    --history_column history \
    --overwrite_cache \
    --model_name_or_path THUDM/chatglm2-6b \
    --output_dir ./test \
    --overwrite_output_dir \
    --max_source_length 1024 \
    --max_target_length 64 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --predict_with_generate \
    --num_train_epochs 2 \
    --logging_steps 20 \
    --save_steps 500 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --report_to tensorboard

