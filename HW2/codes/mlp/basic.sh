conda activate le
export CUDA_VISIBLE_DEVICES=4

SEED=2022
LR=1e-3
HS=1024
DR=0.5
WD=1e-3
NE=100
BS=100
NAME="MLP_${LR}_${DR}_${HS}_${WD}_${BS}_${NE}"
DATASET='../../cifar-10_data'

# Turn off wandb
export WANDB_MODE=offline

function train() {
     python main.py \
        --batch_size $BS \
        --num_epochs $NE \
        --weight_decay $WD \
        --learning_rate $LR \
        --drop_rate $DR \
        --hidden_size $HS \
        --is_train \
        --seed $SEED \
        --name "$NAME" \
        --data_dir $DATASET
}