MODEL="bert-base-uncased" #No trailing / !!
TOKENIZER="bert-base-uncased"

python src/evaluation/lama_probe.py \
  --model_name_or_path $MODEL \
  --tokenizer_name $TOKENIZER \
  --gpu 0 \
  --relations IsA UsedFor AtLocation \
  --full_eval
  #--use_adapter \
  #--use_fusion \
  #--adapter_fusion_path "./adapters/fusion" \
  #--adapter_list "./adapters/isA/" "./adapters/usedFor/" "./adapters/atLocation/" \
