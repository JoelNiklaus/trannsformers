
python3 run_ner.py \
  --model_name_or_path distilbert-base-german-cased \
  --dataset_name joelito/ler \
  --output_dir ./ler/distilbert-base-german-cased \
  --do_train \
  --do_eval \
  --do_predict \
  --save_steps 1000 \
  --eval_steps 100 \
  --seed 42 \
  --per_device_train_batch_size 12