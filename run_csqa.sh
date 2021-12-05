MODEL="./models/mlm_100k_cased" #No trailing / !!
TOKENIZER="bert-base-cased"
DATASET_NAME=commonsense_qa

python src/run_csqa.py \
  --model_name_or_path $MODEL \
  --tokenizer_name $TOKENIZER \
  --max_seq_length 128 \
  --batch_size 32 \
  --learning_rate 1e-5 \
  --num_train_epochs 3 \
  --tune_both True \
