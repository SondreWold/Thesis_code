TRAIN_FILE="./data/concept_net/corpus.txt"
VAL_FILE="./data/concept_net/corpus_val.txt"
TOKENIZER="bert-base-uncased"
MODEL_TYPE="bert-base-uncased"
TRAINING_FOLDER="./src"
OUTPUT_DIR="./models/houlsby_original_uncased"

python $TRAINING_FOLDER/run_mlm.py \
    --model_name_or_path $MODEL_TYPE \
    --train_file $TRAIN_FILE \
    --validation_file $VAL_FILE \
    --output_dir $OUTPUT_DIR \
    --pad_to_max_length \
    --line_by_line True \
    --adapter_name "mlm_houlsby_og" \
    --adapter_config "houlsby" \
    --non_linearity "gelu" \
    --reduction_factor 12      \
    --num_warmup_steps 10000 \
    --max_train_steps 100000 \
