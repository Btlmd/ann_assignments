set -x

# Environment Setup
conda activate fsbm0905

# Random Seed 
SEED=2022

# Disable report to wandb
# OP=' --dr'
OP=''

# Single Hidden Layer
python run_mlp.py --seed $SEED --layout 784 128 10 --loss ce --nonlinearity Relu --name h1_ce_relu $OP 
python run_mlp.py --seed $SEED --layout 784 128 10 --loss mse --nonlinearity Relu --name h1_mse_relu $OP 
python run_mlp.py --seed $SEED --layout 784 128 10 --loss hinge --nonlinearity Relu --name h1_hinge_relu $OP 

python run_mlp.py --seed $SEED --layout 784 128 10 --loss ce --nonlinearity Gelu --name h1_ce_gelu $OP 
python run_mlp.py --seed $SEED --layout 784 128 10 --loss mse --nonlinearity Gelu --name h1_mse_gelu $OP 
python run_mlp.py --seed $SEED --layout 784 128 10 --loss hinge --nonlinearity Gelu --name h1_hinge_gelu $OP 

python run_mlp.py --seed $SEED --layout 784 128 10 --loss ce --nonlinearity Sigmoid --name h1_ce_sigmoid $OP 
python run_mlp.py --seed $SEED --layout 784 128 10 --loss mse --nonlinearity Sigmoid --name h1_mse_sigmoid $OP 
python run_mlp.py --seed $SEED --layout 784 128 10 --loss hinge --nonlinearity Sigmoid --name h1_hinge_sigmoid $OP 

# Double Hidden Layer
python run_mlp.py --seed $SEED --layout 784 256 128 10 --loss ce --nonlinearity Relu --name h2_ce_relu $OP 
python run_mlp.py --seed $SEED --layout 784 256 128 10 --loss mse --nonlinearity Relu --name h2_mse_relu $OP 
python run_mlp.py --seed $SEED --layout 784 256 128 10 --loss hinge --nonlinearity Relu --name h2_hinge_relu $OP 

python run_mlp.py --seed $SEED --layout 784 256 128 10 --loss ce --nonlinearity Gelu --name h2_ce_gelu $OP 
python run_mlp.py --seed $SEED --layout 784 256 128 10 --loss mse --nonlinearity Gelu --name h2_mse_gelu $OP 
python run_mlp.py --seed $SEED --layout 784 256 128 10 --loss hinge --nonlinearity Gelu --name h2_hinge_gelu $OP 

python run_mlp.py --seed $SEED --layout 784 256 128 10 --loss ce --nonlinearity Sigmoid --name h2_ce_sigmoid $OP 
python run_mlp.py --seed $SEED --layout 784 256 128 10 --loss mse --nonlinearity Sigmoid --name h2_mse_sigmoid $OP 
python run_mlp.py --seed $SEED --layout 784 256 128 10 --loss hinge --nonlinearity Sigmoid --name h2_hinge_sigmoid $OP 

# Learning Rate
python run_mlp.py --seed $SEED --layout 784 128 10 --loss ce --nonlinearity Relu --name lr_2e-1 --learning_rate 2e-1 $OP 
python run_mlp.py --seed $SEED --layout 784 128 10 --loss ce --nonlinearity Relu --name lr_1e-1 --learning_rate 1e-1 $OP 
python run_mlp.py --seed $SEED --layout 784 128 10 --loss ce --nonlinearity Relu --name lr_5e-2 --learning_rate 5e-2 $OP 
python run_mlp.py --seed $SEED --layout 784 128 10 --loss ce --nonlinearity Relu --name lr_1e-2 --learning_rate 1e-2 $OP 
python run_mlp.py --seed $SEED --layout 784 128 10 --loss ce --nonlinearity Relu --name lr_5e-3 --learning_rate 5e-3 $OP 
python run_mlp.py --seed $SEED --layout 784 128 10 --loss ce --nonlinearity Relu --name lr_1e-3 --learning_rate 1e-3 $OP 
python run_mlp.py --seed $SEED --layout 784 128 10 --loss ce --nonlinearity Relu --name lr_5e-4 --learning_rate 5e-4 $OP 
python run_mlp.py --seed $SEED --layout 784 128 10 --loss ce --nonlinearity Relu --name lr_1e-4 --learning_rate 1e-4 $OP 
python run_mlp.py --seed $SEED --layout 784 128 10 --loss ce --nonlinearity Relu --name lr_5e-5 --learning_rate 5e-5 $OP 
python run_mlp.py --seed $SEED --layout 784 128 10 --loss ce --nonlinearity Relu --name lr_1e-5 --learning_rate 1e-5 $OP 

# Weight Decay
python run_mlp.py --seed $SEED --layout 784 128 10 --loss ce --nonlinearity Relu --name wd_1e-1 --weight_decay 1e-1 $OP 
python run_mlp.py --seed $SEED --layout 784 128 10 --loss ce --nonlinearity Relu --name wd_1e-2 --weight_decay 1e-2 $OP 
python run_mlp.py --seed $SEED --layout 784 128 10 --loss ce --nonlinearity Relu --name wd_1e-3 --weight_decay 1e-3 $OP 
python run_mlp.py --seed $SEED --layout 784 128 10 --loss ce --nonlinearity Relu --name wd_1e-4 --weight_decay 1e-4 $OP 
python run_mlp.py --seed $SEED --layout 784 128 10 --loss ce --nonlinearity Relu --name wd_1e-5 --weight_decay 1e-5 $OP 
python run_mlp.py --seed $SEED --layout 784 128 10 --loss ce --nonlinearity Relu --name wd_0 --weight_decay 0 $OP 

# Momentum $OP 
python run_mlp.py --seed $SEED --layout 784 128 10 --loss ce --nonlinearity Relu --name mm_0 --momentum 0 $OP 
python run_mlp.py --seed $SEED --layout 784 128 10 --loss ce --nonlinearity Relu --name mm_0.3 --momentum 0.3 $OP 
python run_mlp.py --seed $SEED --layout 784 128 10 --loss ce --nonlinearity Relu --name mm_0.5 --momentum 0.5 $OP 
python run_mlp.py --seed $SEED --layout 784 128 10 --loss ce --nonlinearity Relu --name mm_0.7 --momentum 0.7 $OP 
python run_mlp.py --seed $SEED --layout 784 128 10 --loss ce --nonlinearity Relu --name mm_0.9 --momentum 0.9 $OP 
python run_mlp.py --seed $SEED --layout 784 128 10 --loss ce --nonlinearity Relu --name mm_0.95 --momentum 0.95 $OP 
python run_mlp.py --seed $SEED --layout 784 128 10 --loss ce --nonlinearity Relu --name mm_0.97 --momentum 0.97 $OP 

# Hidden Size
python run_mlp.py --seed $SEED --layout 784 10 --loss ce --nonlinearity Relu --name h_none $OP 
python run_mlp.py --seed $SEED --layout 784 16 10 --loss ce --nonlinearity Relu --name h_16 $OP 
python run_mlp.py --seed $SEED --layout 784 32 10 --loss ce --nonlinearity Relu --name h_32 $OP 
python run_mlp.py --seed $SEED --layout 784 64 10 --loss ce --nonlinearity Relu --name h_64 $OP 
python run_mlp.py --seed $SEED --layout 784 128 10 --loss ce --nonlinearity Relu --name h_128 $OP 
python run_mlp.py --seed $SEED --layout 784 256 10 --loss ce --nonlinearity Relu --name h_256 $OP 
python run_mlp.py --seed $SEED --layout 784 512 10 --loss ce --nonlinearity Relu --name h_512 $OP 
python run_mlp.py --seed $SEED --layout 784 1024 10 --loss ce --nonlinearity Relu --name h_1024 $OP 
python run_mlp.py --seed $SEED --layout 784 2048 10 --loss ce --nonlinearity Relu --name h_2048 $OP 

# Batch Size
python run_mlp.py --seed $SEED --layout 784 128 10 --loss ce --nonlinearity Relu --name bs_1e4 --batch_size 10000 --disp_freq 1 --log epoch $OP 
python run_mlp.py --seed $SEED --layout 784 128 10 --loss ce --nonlinearity Relu --name bs_1e3 --batch_size 1000 --disp_freq 10 --log epoch $OP 
python run_mlp.py --seed $SEED --layout 784 128 10 --loss ce --nonlinearity Relu --name bs_1e2 --batch_size 100 --disp_freq 100 --log epoch $OP 
python run_mlp.py --seed $SEED --layout 784 128 10 --loss ce --nonlinearity Relu --name bs_2e1 --batch_size 20 --disp_freq 500 --log epoch $OP 
python run_mlp.py --seed $SEED --layout 784 128 10 --loss ce --nonlinearity Relu --name bs_1e1 --batch_size 10 --disp_freq 1000 --log epoch $OP 

# Activation
python run_mlp.py --seed $SEED --layout 784 128 10 --loss ce --nonlinearity Gelu --name act_ce_gelu --activation_report --max_epoch 20 $OP 
python run_mlp.py --seed $SEED --layout 784 128 10 --loss mse --nonlinearity Gelu --name act_mse_gelu --activation_report --max_epoch 20 $OP 
python run_mlp.py --seed $SEED --layout 784 128 10 --loss hinge --nonlinearity Gelu --name act_hinge_gelu --activation_report --max_epoch 20 $OP 

python run_mlp.py --seed $SEED --layout 784 128 10 --loss ce --nonlinearity Relu --name act_ce_relu --activation_report --max_epoch 20 $OP 
python run_mlp.py --seed $SEED --layout 784 128 10 --loss mse --nonlinearity Relu --name act_mse_relu --activation_report --max_epoch 20 $OP 
python run_mlp.py --seed $SEED --layout 784 128 10 --loss hinge --nonlinearity Relu --name act_hinge_relu --activation_report --max_epoch 20 $OP 

python run_mlp.py --seed $SEED --layout 784 256 128 10 --loss ce --nonlinearity Relu --name act2_ce_relu --activation_report --max_epoch 20 $OP 
python run_mlp.py --seed $SEED --layout 784 256 128 10 --loss mse --nonlinearity Relu --name act2_mse_relu --activation_report --max_epoch 20 $OP 
python run_mlp.py --seed $SEED --layout 784 256 128 10 --loss hinge --nonlinearity Relu --name act2_hinge_relu --activation_report --max_epoch 20 $OP 

python run_mlp.py --seed $SEED --layout 784 256 128 10 --loss ce --nonlinearity Gelu --name act2_ce_gelu --activation_report --max_epoch 20 $OP 
python run_mlp.py --seed $SEED --layout 784 256 128 10 --loss mse --nonlinearity Gelu --name act2_mse_gelu --activation_report --max_epoch 20 $OP 
python run_mlp.py --seed $SEED --layout 784 256 128 10 --loss hinge --nonlinearity Gelu --name act2_hinge_gelu --activation_report --max_epoch 20 $OP 

# Gradient Norm 
python run_mlp.py --seed $SEED --layout 784 128 10 --loss ce --nonlinearity Gelu --name norm_ce_gelu --norm_report --max_epoch 20 $OP 
python run_mlp.py --seed $SEED --layout 784 128 10 --loss mse --nonlinearity Gelu --name norm_mse_gelu --norm_report --max_epoch 20 $OP 
python run_mlp.py --seed $SEED --layout 784 128 10 --loss hinge --nonlinearity Gelu --name norm_hinge_gelu --norm_report --max_epoch 20 $OP 

python run_mlp.py --seed $SEED --layout 784 128 10 --loss ce --nonlinearity Relu --name norm_ce_relu --norm_report --max_epoch 20 $OP 
python run_mlp.py --seed $SEED --layout 784 128 10 --loss mse --nonlinearity Relu --name norm_mse_relu --norm_report --max_epoch 20 $OP 
python run_mlp.py --seed $SEED --layout 784 128 10 --loss hinge --nonlinearity Relu --name norm_hinge_relu --norm_report --max_epoch 20 $OP 

python run_mlp.py --seed $SEED --layout 784 128 10 --loss ce --nonlinearity Sigmoid --name norm_ce_sigmoid --norm_report --max_epoch 20 $OP 
python run_mlp.py --seed $SEED --layout 784 128 10 --loss mse --nonlinearity Sigmoid --name norm_mse_sigmoid --norm_report --max_epoch 20 $OP 
python run_mlp.py --seed $SEED --layout 784 128 10 --loss hinge --nonlinearity Sigmoid --name norm_hinge_sigmoid --norm_report --max_epoch 20 $OP 

python run_mlp.py --seed $SEED --layout 784 256 128 10 --loss ce --nonlinearity Relu --name norm2_ce_relu --norm_report --max_epoch 20 $OP 
python run_mlp.py --seed $SEED --layout 784 256 128 10 --loss mse --nonlinearity Relu --name norm2_mse_relu --norm_report --max_epoch 20 $OP 
python run_mlp.py --seed $SEED --layout 784 256 128 10 --loss hinge --nonlinearity Relu --name norm2_hinge_relu --norm_report --max_epoch 20 $OP 

python run_mlp.py --seed $SEED --layout 784 256 128 10 --loss ce --nonlinearity Gelu --name norm2_ce_gelu --norm_report --max_epoch 20 $OP 
python run_mlp.py --seed $SEED --layout 784 256 128 10 --loss mse --nonlinearity Gelu --name norm2_mse_gelu --norm_report --max_epoch 20 $OP 
python run_mlp.py --seed $SEED --layout 784 256 128 10 --loss hinge --nonlinearity Gelu --name norm2_hinge_gelu --norm_report --max_epoch 20 $OP 

python run_mlp.py --seed $SEED --layout 784 256 128 10 --loss ce --nonlinearity Sigmoid --name norm2_ce_sigmoid --norm_report --max_epoch 20 $OP 
python run_mlp.py --seed $SEED --layout 784 256 128 10 --loss mse --nonlinearity Sigmoid --name norm2_mse_sigmoid --norm_report --max_epoch 20 $OP 
python run_mlp.py --seed $SEED --layout 784 256 128 10 --loss hinge --nonlinearity Sigmoid --name norm2_hinge_sigmoid --norm_report --max_epoch 20 $OP 
