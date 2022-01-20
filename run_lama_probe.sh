MODEL="bert-base-uncased" #No trailing / !!
TOKENIZER="bert-base-uncased"
DATASET_NAME=commonsense_qa

python src/evaluation/lama_probe.py \
  --model_name_or_path $MODEL \
  --tokenizer_name $TOKENIZER \
  --gpu 0 \ 
  --at_k 5
