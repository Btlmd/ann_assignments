#!/usr/bin/env bash

set -x

DEVICE=' --device cuda:0 '

# Tfmr-scratch v. Tfmr-finetune
python main.py --name Tfmr_scratch_raw --pretrain_dir None $DEVICE
python main.py --name Tfmr_finetune_raw --pretrain_dir ../pretrain $DEVICE
python main.py --name Tfmr_scratch_shuffle --pretrain_dir None --shuffle $DEVICE
python main.py --name Tfmr_finetune_shuffle --pretrain_dir ../pretrain --shuffle $DEVICE

# Another Criteria
python main.py --name Tfmr_scratch_shuffle_dev_bleu --es_criteria bleu --pretrain_dir None --tolerance 2 --shuffle $DEVICE
python main.py --name Tfmr_finetune_shuffle_dev_bleu --es_criteria bleu --pretrain_dir ../pretrain --tolerance 2 --shuffle $DEVICE
python main.py --name Tfmr_scratch_shuffle_dev_bleu_3 --es_criteria bleu --pretrain_dir None --tolerance 3 --shuffle $DEVICE
python main.py --name Tfmr_finetune_shuffle_dev_bleu_3 --es_criteria bleu --pretrain_dir ../pretrain --tolerance 3 --shuffle $DEVICE

# Finetune 12 layers v. 3 layers
python main.py --name 12_layers --layers 1 2 3 4 5 6 7 8 9 10 11 12 --pretrain_dir ../pretrain --shuffle $DEVICE
python main.py --name 3_layers --layers 1 2 3 --pretrain_dir ../pretrain --shuffle $DEVICE

# Finetune layer selection
python main.py --name layer_01_02_03 --layers  1  2  3 --pretrain_dir ../pretrain --shuffle $DEVICE
python main.py --name layer_05_06_07 --layers  5  6  7 --pretrain_dir ../pretrain --shuffle $DEVICE
python main.py --name layer_10_11_12 --layers 10 11 12 --pretrain_dir ../pretrain --shuffle $DEVICE
python main.py --name layer_01_06_12 --layers  1  6 12 --pretrain_dir ../pretrain --shuffle $DEVICE

# Scratch Head Number
for N_HEAD in 1 3 6 12 24 48 96 192
do
  python main.py --name head_$N_HEAD --head $N_HEAD --shuffle --device cuda:$G &
done