MODEL="./models/mnli" #No trailing / !!
TOKENIZER="bert-base-uncased"
ADAPTER_NAME="mlm_houlsby"
OUTPUT_DIR="./models/mnli"
python -u ./src/evaluation/run_glue_diagnostic.py \
  --model_name_or_path $MODEL \
  --adapter_name $ADAPTER_NAME \
  --tokenizer_name $TOKENIZER \
  --output_dir $OUTPUT_DIR \
  --use_adapter True \
