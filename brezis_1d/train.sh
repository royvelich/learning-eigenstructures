NAME="AnchorPoints_20percent_reconstruction_laplacian"
SIGNAL_LENGTH=100
SMOOTHING_SIGMA=3
K=5
SIGNALS_PER_EPOCH=1_000_000
HIDDEN_DIM=256
HIDDEN_LAYERS=5
MAX_EPOCHS=100
NUM_WORKERS=16
WD=0.00001

LR=(0.00001)
BATCH_SIZE=(4096)
ACTIVATIONS=("relu")

for ACTIVATION in "${ACTIVATIONS[@]}"; do
    for BS in "${BATCH_SIZE[@]}"; do
        for LEARNING_RATE in "${LR[@]}"; do
            RUN_NAME="${NAME}_${SIGNAL_LENGTH}_activation_${ACTIVATION}_lr_${LEARNING_RATE}_wd_${WD}_bs_${BS}_max_epochs_${MAX_EPOCHS}_k_${K}_hidden_dim${HIDDEN_DIM}_hidden_layers_${HIDDEN_LAYERS}"
            # Run the Python command with the current fold values
            python src/train.py \
                --name $RUN_NAME \
                --signal_length $SIGNAL_LENGTH \
                --activation $ACTIVATION \
                --smoothing_sigma $SMOOTHING_SIGMA \
                --k $K \
                --signals_per_epoch $SIGNALS_PER_EPOCH \
                --hidden_dim $HIDDEN_DIM \
                --hidden_layers $HIDDEN_LAYERS \
                --max_epochs $MAX_EPOCHS \
                --num_workers $NUM_WORKERS \
                --batch_size $BS \
                --lr $LEARNING_RATE \
                --weight_decay $WD
        done
    done
done