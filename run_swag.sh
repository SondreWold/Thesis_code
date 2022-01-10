MODEL="distilbert-base-uncased" #No trailing / !!
TOKENIZER="distilbert-base-uncased"
DATASET_NAME=swag

python src/run_swag.py \
  --model_name_or_path $MODEL \
  --tokenizer_name $TOKENIZER \
  --dataset_name $DATASET_NAME \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
