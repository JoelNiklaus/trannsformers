import os

from util import make_reproducible, compute_metrics

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # do this to remove gpu with full memory (MUST be before torch import)

from pprint import pprint

from transformers import DistilBertTokenizerFast, EvaluationStrategy
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments

import torch

from datasets import load_dataset

seed = 42
make_reproducible(seed)

os.environ["TOKENIZERS_PARALLELISM"] = "true"  # used for disabling warning (BUT: if deadlock occurs, remove this)

base_model_path = "distilbert-base-uncased"
model_path = "./results/checkpoint-2500/"
dataset_path = "joelito/sem_eval_2010_task_8"

print("Loading Dataset")
dataset = load_dataset(dataset_path)
num_labels = dataset['train'].features['relation'].num_classes

idxToLabelsList = dataset['train'].features['relation'].names  # list to look up the label indices
id2label = {k: v for k, v in enumerate(idxToLabelsList)}
label2id = {v: k for k, v in enumerate(idxToLabelsList)}

print("Loading Model")
model = DistilBertForSequenceClassification.from_pretrained(model_path, num_labels=num_labels, id2label=id2label,
                                                            label2id=label2id)
print("Loading Tokenizer")
tokenizer = DistilBertTokenizerFast.from_pretrained(base_model_path)

print("Tokenizing Dataset")
supervised_keys = dataset['train'].supervised_keys  # 'sentence' and 'relation'
dataset = dataset.map(lambda ex: tokenizer(ex[supervised_keys.input], truncation=True, padding='max_length'),
                      batched=True)
dataset.rename_column_(original_column_name=supervised_keys.output,
                       new_column_name='label')  # IMPORTANT: otherwise the loss cannot be computed
dataset.set_format(type='pt', columns=['input_ids', 'attention_mask', 'label'], output_all_columns=True)

training_args = TrainingArguments(
    output_dir='./results',  # output directory
    num_train_epochs=10,  # total number of training epochs
    per_device_train_batch_size=10,  # batch size per device during training
    per_device_eval_batch_size=10,  # batch size for evaluation
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir='./logs',  # directory for storing logs
    logging_steps=10,
    save_steps=100,
    eval_steps=100,
    evaluation_strategy=EvaluationStrategy.STEPS,
    seed=seed,
    do_train=False,
    do_eval=False,
    do_predict=True,
)

trainer = Trainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=dataset['train'],  # training dataset
    eval_dataset=dataset['validation'],  # evaluation dataset
    compute_metrics=compute_metrics,  # additional metrics to the loss
)

if training_args.do_train:
    print("Training")
    trainer.train()
    # For convenience, we also re-save the tokenizer to the same directory,
    if trainer.is_world_master():
        tokenizer.save_pretrained(training_args.output_dir)

if training_args.do_eval:
    print("Evaluating on validation set")
    metrics = trainer.evaluate()
    print(metrics)


def get_prediction_ids(predictions):
    tensor = torch.tensor(predictions)
    softmax = torch.nn.functional.softmax(tensor, dim=-1)
    argmax = torch.argmax(softmax, dim=-1)
    return argmax.tolist()


if training_args.do_predict:
    data_sub_set_size = 5

    test_dataset = dataset['test']
    # test_dataset = test_dataset.select(indices=range(data_sub_set_size))  # make faster by only selecting small subset

    print(f"Predicting on {data_sub_set_size} instances of the test set")
    predictions, label_ids, metrics = trainer.predict(test_dataset)
    print(metrics)

    prediction_ids = get_prediction_ids(predictions)  # get ids of predictions
    predicted_labels = [idxToLabelsList[prediction_id] for prediction_id in prediction_ids]  # get labels of predictions
    correct_labels = [idxToLabelsList[label_id] for label_id in label_ids]  # get labels of ground truth

    for i in range(data_sub_set_size):
        print(f"\nSentence: {dataset['test'][i]['sentence']}")
        print(f"Predicted Relation: {predicted_labels[i]}")
        print(f"Ground Truth Relation: {correct_labels[i]}")
