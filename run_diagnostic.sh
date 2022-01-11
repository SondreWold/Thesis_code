MODEL="./models/mlm_100k_cased" #No trailing / !!
TOKENIZER="bert-base-cased"
ADAPTER_NAME=""
OUTPUT_DIR=""
python -u ./src/evaluation/run_glue_diagnostic.py \
  --model_name_or_path $MODEL \
  --adapter_name $ADAPTER_NAME
  --tokenizer_name $TOKENIZER \
  --output_dir $OUTPUT_DIR/ 
