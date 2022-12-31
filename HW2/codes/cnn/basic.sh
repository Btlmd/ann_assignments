conda activate le
export CUDA_VISIBLE_DEVICES=4

SEED=2022
LR=1e-3
DR='0.4 0.4'
WD=1e-5
NE=100
BS=100
CCH='128 512'
CKE='5 7'
PKE='5 5'
PST='3 4'
NAME="CNN_${CCH}_${CKE}_${LR}_${DR}_${WD}_${BS}_${NE}"
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
        --conv_ch $CCH \
        --conv_ker $CKE \
        --pool_ker $PKE \
        --pool_stride $PST \
        --is_train \
        --seed $SEED \
        --name "$NAME" \
        --data_dir $DATASET
}