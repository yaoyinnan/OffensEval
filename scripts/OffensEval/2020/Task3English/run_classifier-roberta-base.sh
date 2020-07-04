#!/usr/bin/env bash
set -eux

export TASK=OffensEval
export TASK_NAME=OffensEval2020Task3English
export DATA_DIR=data/OffensEval/2020/Task3English
export OUTPUT_NAME=output
export MODEL=roberta
export MODEL_NAME=roberta-base
export STAGE_NUM=1
export NEXT_STAGE_NUM=2

python ./examples/run_classifier.py \
    --model_type ${MODEL} \
    --model_name_or_path ${MODEL_NAME} \
    --task_name ${TASK_NAME} \
    --do_lower_case \
    --data_dir ${DATA_DIR} \
    --max_seq_length 128 \
    --per_gpu_train_batch_size 4   \
    --per_gpu_eval_batch_size 64   \
    --per_gpu_test_batch_size 64   \
    --per_gpu_pred_batch_size 64   \
    --learning_rate 1e-5 \
    --weight_decay 0.0001 \
    --num_train_epochs 10.0 \
    --output_dir ${OUTPUT_NAME}/${TASK}/${TASK_NAME}-${MODEL_NAME}/stage_${NEXT_STAGE_NUM} \
    --save_steps 1000 \
    --overwrite_cache \
    --eval_all_checkpoints \
    --do_test \
    --do_eval \
    --do_train \
#    --do_pred \
