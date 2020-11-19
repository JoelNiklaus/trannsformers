import torch
import numpy as np
import random

from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def make_reproducible(seed=42):
    torch.set_deterministic(True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def compute_metrics(pred, f1_average='micro'):
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
