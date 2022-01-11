MODEL="bert-base-uncased"
TOKENIZER="bert-base-uncased"
TASK_NAME="mnli"
OUTPUT_DIR="./models/mnli/"

python -u src/run_glue.py \
  --model_name_or_path $MODEL \
  --tokenizer_name $TOKENIZER \
  --task_name $TASK_NAME \
  --max_length 128 \
  --per_device_train_batch_size 16 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --output_dir $OUTPUT_DIR \
