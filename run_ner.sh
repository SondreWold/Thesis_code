python src/run_ner.py \
  --model_name_or_path bert-base-cased \
  --tokenizer_name bert-base-cased \
  --dataset_name conll2003 \
  --output_dir /tmp/test-ner \
  --do_train \
  --do_eval