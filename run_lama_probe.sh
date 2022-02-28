MODEL="./models/fusion_test/fusion_test" #No trailing / !!
TOKENIZER="bert-base-uncased"

python src/evaluation/lama_probe.py \
  --model_name_or_path $MODEL \
  --tokenizer_name $TOKENIZER \
  --relations IsA UsedFor AtLocation \
  --use_adapter \
  --use_fusion \
  --at_k 5
