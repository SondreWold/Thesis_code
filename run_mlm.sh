TRAIN_FILE="./data/concept_net/cn_train.txt"
VAL_FILE="./data/concept_net/cn_validation.txt"
MODEL_TYPE="bert-base-uncased"
TRAINING_FOLDER="./src"
OUTPUT_DIR="./models/"

python $TRAINING_FOLDER/run_mlm.py \
    --model_name_or_path $MODEL_TYPE \
    --train_file $TRAIN_FILE \
    --validation_file $VAL_FILE \
    --output_dir $OUTPUT_DIR \
    --line_by_line true \
    --pad_to_max_length \
    --adapter_name "mlm_houlsby" \
    --adapter_config "houlsby" \
    --max_train_steps 10 \
