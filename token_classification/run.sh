
python3 run_ner.py \
  --model_name_or_path distilbert-base-uncased \
  --dataset_name joelito/ler \
  --output_dir ./ler \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size 14