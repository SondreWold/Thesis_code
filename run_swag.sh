MODEL="./models/mlm_100k_cased" #No trailing / !!
TOKENIZER="bert-base-cased"
DATASET_NAME=swag

python src/run_swag_qa.py \
  --model_name_or_path $MODEL \
  --tokenizer_name $TOKENIZER \
  --dataset_name $DATASET_NAME \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --tune_both True \
  --output_dir /tmp/$DATASET_NAME/