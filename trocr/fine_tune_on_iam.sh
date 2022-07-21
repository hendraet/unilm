#!/bin/sh

cd unilm/trocr

export MODEL_NAME=ft_iam_debug
export SAVE_PATH=/models${MODEL_NAME}
export LOG_DIR=logs/log_${MODEL_NAME}
export DATA=/datasets/trocr_iam
mkdir -p ${LOG_DIR}
export BSZ=8
export valid_BSZ=16
export PRETRAINED_MODEL=/models/trocr-base-stage1.pt

 # --bpe sentencepiece --sentencepiece-model ./unilm3-cased.model --decoder-pretrained unilm ## For small models
#CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 \
#python3 -m torch.distributed.launch --nproc_per_node=1 \
python3 $(which fairseq-train) \
    --data-type STR --user-dir ./ --task text_recognition \
    --arch trocr_base \
    --seed 1111 --optimizer adam --lr 2e-05 --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-8 --warmup-updates 500 --weight-decay 0.0001 --log-format tqdm \
    --log-interval 10 --batch-size ${BSZ} --batch-size-valid ${valid_BSZ} --save-dir ${SAVE_PATH} \
    --tensorboard-logdir ${LOG_DIR} --max-epoch 300 --patience 20 --ddp-backend legacy_ddp \
    --num-workers 8 --preprocess DA2 --update-freq 1 \
    --bpe gpt2 --decoder-pretrained roberta \
    --finetune-from-model ${PRETRAINED_MODEL} --fp16 \
    ${DATA}
