from transformers import TFDistilBertForSequenceClassification

checkpoint_dir = "/home/joel/transformers/sem_eval_2010_task_8/distilbert-base-uncased-local/results/checkpoint-3500"

tf_model = TFDistilBertForSequenceClassification.from_pretrained(checkpoint_dir, from_pt=True)
tf_model.save_pretrained(checkpoint_dir)
