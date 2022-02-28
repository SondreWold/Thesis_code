MODEL="roberta-base" #No trailing / !!
TOKENIZER="roberta-base"
DATASET_NAME=commonsense_qa

python src/run_csqa.py \
  --model_name_or_path $MODEL \
  --tokenizer_name $TOKENIZER \
  --max_seq_length 128 \
  --pad_to_max_length \
  --model_type "roberta" \
  --batch_size 1 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
