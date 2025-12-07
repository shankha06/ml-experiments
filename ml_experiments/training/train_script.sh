# Configuration
NUM_GPUS=4
MODEL="BAAI/bge-base-en-v1.5"
INPUT_PARQUET="data/mnrl_training_data.parquet"
TRAIN_DATA="data/mnrl_training_data.jsonl"
OUTPUT_DIR="models/bge-finetuned-mnrl"
BATCH_SIZE=20
EPOCHS=3

# Ensure output directory exists
mkdir -p $OUTPUT_DIR

# 1. Convert Data
echo "Converting data to BGE format..."
python src/convert_to_bge.py --input "$INPUT_PARQUET" --output "$TRAIN_DATA"

echo "Starting FlagEmbedding training on $NUM_GPUS GPUs..."

# 2. Launch FlagEmbedding
# Using the module directly
torchrun --nproc_per_node=$NUM_GPUS \
    -m FlagEmbedding.baai_general_embedding.finetune.run \
    --output_dir "$OUTPUT_DIR" \
    --model_name_or_path "BAAI/bge-base-en-v1.5" \
    --train_data "$TRAIN_DATA" \
    --learning_rate 2e-5 \
    --fp16 \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps 4 \
    --dataloader_drop_last True \
    --normlized True \
    --temperature 0.02 \
    --query_max_len 64 \
    --passage_max_len 256 \
    --train_group_size 5 \
    --negatives_cross_device \
    --logging_steps 10 \
    --save_strategy epoch \
    --query_instruction_for_retrieval "Represent this sentence for searching relevant passages: "

echo "Training complete. Model saved to $OUTPUT_DIR"
