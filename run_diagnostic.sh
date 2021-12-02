MODEL="./models/mlm_100k_cased" #No trailing / !!
TOKENIZER="bert-base-cased"
python -u ./src/evaluation/run_glue_diagnostic.py \
  --model_name_or_path $MODEL \
  --tokenizer_name $TOKENIZER \
  --output_dir $MODEL/ 
