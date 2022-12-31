# Dropout Rate
source ./basic.sh
for ADR in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
    DR="${ADR} ${ADR}"
    NAME="DR_${ADR}"
    train
done

# Batch Size
source ./basic.sh
for BS in 5 50 100 500
do
    NAME="BS_${BS}"
    train
done

# Learning Rate
source ./basic.sh
for LR in 1e-1 1e-2 1e-3 1e-4 1e-5 
do
    NAME="LR_${LR}"
    train
done