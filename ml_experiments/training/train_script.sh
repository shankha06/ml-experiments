#!/bin/bash
set -e

# Configuration
NUM_GPUS=4
MODEL_NAME="BAAI/bge-base-en-v1.5"
INPUT_RAW="data/faq_dataset.parquet"
TRAIN_DATA="data/mnrl_training_data.parquet"
OUTPUT_DIR="models/bge-finetuned-mnrl"
BATCH_SIZE=32 # Per GPU, total effective = 32*4 = 128
EPOCHS=3
LR=2e-5

# Ensure output directory exists
mkdir -p $(dirname $OUTPUT_DIR)

# 1. Prepare Data (if not exists)
if [ ! -f "$TRAIN_DATA" ]; then
    echo "Files not found, generating training data..."
    # Check if raw data exists
    if [ ! -f "$INPUT_RAW" ]; then
        echo "Error: Raw input $INPUT_RAW not found!"
        # Try to generate dummy data for testing if raw not found?
        # Assuming user has data or will create it.
        # But wait, looking at file list, src/create_dummy_data.py exists.
        echo "Attempting to create dummy data..."
        python src/create_dummy_data.py --output "$INPUT_RAW"
    fi
    python src/prepare_data.py --input "$INPUT_RAW" --output "$TRAIN_DATA"
fi

echo "Starting NATIVE DDP training on $NUM_GPUS GPUs..."

# 2. Launch Training
torchrun --nproc_per_node=$NUM_GPUS \
    src/train.py \
    --input "$TRAIN_DATA" \
    --output "$OUTPUT_DIR" \
    --model_name "$MODEL_NAME" \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --accumulation_steps 1 \
    --num_negatives 5

echo "Training complete. Model saved to $OUTPUT_DIR"

