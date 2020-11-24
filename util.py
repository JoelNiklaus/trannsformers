import torch
import numpy as np
import random

from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def get_prediction_ids(predictions):
    """ Get the label ids from the raw prediction outputs """
    tensor = torch.tensor(predictions)
    softmax = torch.nn.functional.softmax(tensor, dim=-1)
    argmax = torch.argmax(softmax, dim=-1)
    return argmax.tolist()


def make_reproducible(seed=42):
    """ Make the run reproducible by setting the seed """
    torch.set_deterministic(True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def compute_metrics(pred, f1_average='weighted'):
    """ Defining additional metrics to be computed """
    labels = pred.label_ids
    predictions = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average=f1_average)
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
