python src/run_ner.py \
  --model_name_or_path ./models/houlsby_original_cased \
  --tokenizer_name bert-base-cased \
  --dataset_name conll2003 \
  --output_dir /tmp/test-ner \
  --do_train \
  --tune_both True \
  --adapter_name mlm_houlsby
  --do_eval