#!/usr/bin/env bash

set -x

# latent_dim & hidden_dim
for SD in 2022 2023 2024 2025 2026
do
  for LD in 16 32 64 100
  do
    for HD in 16 32 64 100
    do
      echo "[Latent Dimension $LD, Hidden Dimension $HD, Seed $SD]"
      CUDA_VISIBLE_DEVICES=0 python main.py \
        --do_train \
        --latent_dim $LD \
        --generator_hidden_dim $HD \
        --discriminator_hidden_dim $HD \
        --seed $SD \
        --num_training_steps 5000
    done
  done
done

# MLP latent_dim
for SD in 2022 2023 2024 2025 2026
do
  for LD in 16 32 64 100
  do
    ARCH_G="$LD 128 256 512 1024"
    ARCH_D="1024 512 256 128 1"
    echo "[Latent Dimension $LD, Seed $SD]"
    CUDA_VISIBLE_DEVICES=0 python main.py \
    --do_train \
    --mlp \
    --mlp_g_arch $ARCH_G \
    --mlp_d_arch $ARCH_D \
    --seed $SD \
    --num_training_steps 10000
  done
done

# CNN
EXP='--latent_dim 16 --generator_hidden_dim 100 --discriminator_hidden_dim 100 --seed 2026'
LERP='--interpolation_batch 10 --interpolation_K 20'

# interpolation
CUDA_VISIBLE_DEVICES=0 python main.py $LERP $EXP --interpolation_range 0 1
CUDA_VISIBLE_DEVICES=0 python main.py $LERP $EXP --interpolation_range -1 2
CUDA_VISIBLE_DEVICES=0 python main.py $LERP $EXP --interpolation_range -10 11

# sampling to check feature collapse
CUDA_VISIBLE_DEVICES=0 python main.py --sampling 100 $EXP

# MLP
ARCH_G="100 128 256 512 1024"
ARCH_D="1024 512 256 128 1"
EXP="--mlp --mlp_g_arch $ARCH_G --mlp_d_arch $ARCH_D --seed 2022 --num_training_steps 10000"
LERP='--interpolation_batch 10 --interpolation_K 20'

# interpolation
CUDA_VISIBLE_DEVICES=0 python main.py $LERP $EXP --interpolation_range 0 1
CUDA_VISIBLE_DEVICES=0 python main.py $LERP $EXP --interpolation_range -1 2
CUDA_VISIBLE_DEVICES=0 python main.py $LERP $EXP --interpolation_range -10 11

# sampling to check feature collapse
CUDA_VISIBLE_DEVICES=0 python main.py --sampling 100 $EXP
