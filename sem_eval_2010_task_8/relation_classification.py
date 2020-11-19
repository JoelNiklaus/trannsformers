import os
import random
import fire

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # do this to remove gpu with full memory (MUST be before torch import)
os.environ["TOKENIZERS_PARALLELISM"] = "true"  # used for disabling warning (BUT: if deadlock occurs, remove this)

from transformers import EvaluationStrategy, AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments

from datasets import load_dataset

from util import make_reproducible, compute_metrics, get_prediction_ids

from pprint import pprint


def run(base_model="roberta-base", fine_tuned_checkpoint_name=None, dataset="joelito/sem_eval_2010_task_8",
        do_train=False, do_eval=False, do_predict=True, test_set_sub_size=None, seed=42):
    """
    Runs the specified transformer model
    :param base_model:             the name of the base model from huggingface transformers (e.g. roberta-base)
    :param fine_tuned_checkpoint_name:  the name of the fine tuned checkpoint (e.g. checkpoint-500)
    :param dataset:                the name of the dataset from huggingface datasets (e.g. joelito/sem_eval_2010_task_8)
    :param do_train:                    whether to train the model
    :param do_eval:                     whether to evaluate the model in the end
    :param do_predict:                  whether to do predictions on the test set in the end
    :param test_set_sub_size:           make faster by only selecting small subset, otherwise just set to False/None
    :param seed:                        random seed for reproducibility
    :return:
    """

    dir_path = os.path.dirname(os.path.realpath(__file__))
    local_model_name = f"{dir_path}/{base_model}-local"

    make_reproducible(seed)
    training_args = TrainingArguments(
        output_dir=f'{local_model_name}/results',  # output directory
        num_train_epochs=10,  # total number of training epochs
        per_device_train_batch_size=10,  # batch size per device during training
        per_device_eval_batch_size=10,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir=f'{local_model_name}/logs',  # directory for storing logs
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        evaluation_strategy=EvaluationStrategy.STEPS,
        seed=seed,
    )

    print("Loading Dataset")
    dataset = load_dataset(dataset)
    num_labels = dataset['train'].features['relation'].num_classes

    idx_to_labels_list = dataset['train'].features['relation'].names  # list to look up the label indices
    id2label = {k: v for k, v in enumerate(idx_to_labels_list)}
    label2id = {v: k for k, v in enumerate(idx_to_labels_list)}

    model_path = base_model
    if fine_tuned_checkpoint_name:
        model_path = f"{local_model_name}/{training_args.output_dir}/{fine_tuned_checkpoint_name}"

    print("Loading Model")
    model = AutoModelForSequenceClassification.from_pretrained(model_path, id2label=id2label, label2id=label2id,
                                                               num_labels=num_labels)
    print("Loading Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("Tokenizing Dataset")
    supervised_keys = dataset['train'].supervised_keys  # 'sentence' and 'relation'
    dataset = dataset.map(lambda ex: tokenizer(ex[supervised_keys.input], truncation=True, padding='max_length'),
                          batched=True)
    dataset.rename_column_(original_column_name=supervised_keys.output,
                           new_column_name='label')  # IMPORTANT: otherwise the loss cannot be computed
    dataset.set_format(type='pt', columns=['input_ids', 'attention_mask', 'label'], output_all_columns=True)

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=dataset['train'],  # training dataset
        eval_dataset=dataset['validation'],  # evaluation dataset
        compute_metrics=compute_metrics,  # additional metrics to the loss
    )

    if do_train:
        print("Training on train set")
        trainer.train()
        # For convenience, we also re-save the tokenizer to the same directory,
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    if do_eval:
        print("Evaluating on validation set")
        metrics = trainer.evaluate()
        print(metrics)

    if do_predict:
        print(f"Predicting on test set")
        if test_set_sub_size:
            # IMPORTANT: This command somehow may delete some features in the dataset!
            dataset['test'] = dataset['test'].select(indices=range(test_set_sub_size))

        sentences = dataset['test'][0:-1]['sentence'] # save sentences because they will be removed by trainer.predict()

        predictions, label_ids, metrics = trainer.predict(dataset['test'])
        print(metrics)

        prediction_ids = get_prediction_ids(predictions)  # get ids of predictions
        predicted_labels = [idx_to_labels_list[prediction_id] for prediction_id in
                            prediction_ids]  # get labels of predictions
        correct_labels = [idx_to_labels_list[label_id] for label_id in label_ids]  # get labels of ground truth

        examples = random.sample(range(dataset['test'].num_rows), 5)  # look at five random examples from the dataset
        for i in examples:
            print(f"\nSentence: {sentences[i]}")
            print(f"Predicted Relation: {predicted_labels[i]}")
            print(f"Ground Truth Relation: {correct_labels[i]}")


if __name__ == '__main__':
    fire.Fire(run)
