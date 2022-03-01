MODEL="bert-base-uncased" #No trailing / !!
TOKENIZER="bert-base-uncased"

python src/evaluation/lama_probe.py \
  --model_name_or_path $MODEL \
  --tokenizer_name $TOKENIZER \
  --relations IsA UsedFor AtLocation \
  --use_adapter \
  --use_fusion \
  --adapter_fusion_path "./adapters/fusion" \
  --adapter_list "./adapters/isA/" "./adapters/usedFor/" "./adapters/atLocation/" \
  --full_eval
