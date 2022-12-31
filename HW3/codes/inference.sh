#!/usr/bin/env bash

set -x

DEVICE=' --device cuda:0'

 Tfmr-scratch v. Tfmr-finetune dev_bleu
python main.py --test Tfmr_scratch_shuffle_dev_bleu $DEVICE && python main.py --test Tfmr_finetune_shuffle_dev_bleu $DEVICE

# Evaluate blue score
python main.py --data_cross_bleu

# Tfmr-scratch v. Tfmr-finetune
GID=0
for NAME in Tfmr_scratch_raw Tfmr_scratch_shuffle Tfmr_finetune_raw Tfmr_finetune_shuffle
do
  for TEMP in 0.7 1.0
  do
    for DS in random top-p top-k
    do
      python main.py --test $NAME --temperature $TEMP --decode_strategy $DS --device cuda:$GID
    done
  done
done

GID=0
for NAME in Tfmr_scratch_shuffle_dev_bleu Tfmr_finetune_shuffle_dev_bleu
do
  for TEMP in 0.7 1.0
  do
    for DS in random top-p top-k
    do
      python main.py --test $NAME --temperature $TEMP --decode_strategy $DS --device cuda:$GID
    done
  done
done

DEFAULT_INF=" --decode_strategy random --temperature 0.7 $DEVICE"

# Finetune 12 layers v. 3 layers
for LAYERS in 12_layers 3_layers
do
  python main.py --test $LAYERS $DEFAULT_INF
done

# Number of attention heads
for N_HEAD in 1 3 6 12 24 48 96 192
do
  python main.py --test head_$N_HEAD $DEFAULT_INF
done

# Finetune layer structure
for LAYERS in layer_01_02_03 layer_05_06_07 layer_10_11_12 layer_01_06_12
do
  python main.py --test $LAYERS $DEFAULT_INF
done

# Random Pick Results
python select_output.py --reg_exp 'Tfmr_(scratch|finetune)_shuffle_dev'