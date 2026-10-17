"""
Evaluation metrics required by the assignment, plus Top-5.

All functions take:
  logits - tensor (N, C): the model's raw scores for N images and C classes
  labels - tensor (N,):   the ground-truth class of each image
"""

import torch


def top1_accuracy(logits, labels):
    """Top-1 accuracy = (1/N) * sum( 1{argmax(y) == ground truth} ).

    argmax over the class dimension gives the predicted class of each image.
    Comparing with labels gives True/False per image; the mean of those is the accuracy.
    (argmax of the logits equals argmax of the softmax probabilities, since softmax
    never changes which score is largest.)
    """
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def topk_accuracy(logits, labels, k=5):
    """Fraction of images whose true class is among the k highest-scoring classes."""
    topk = logits.topk(k, dim=1).indices                 # (N, k) best k class indices per image
    hits = (topk == labels.unsqueeze(1)).any(dim=1)       # True if the label is anywhere in the k
    return hits.float().mean().item()


def per_class_accuracy(logits, labels, num_classes):
    """Return a tensor (C,) where entry i is the accuracy on images of class i (T_i)."""
    preds = logits.argmax(dim=1)
    correct = (preds == labels).float()
    # bincount with weights sums 'correct' separately for each class label in one call
    correct_per_class = torch.bincount(labels, weights=correct, minlength=num_classes)
    total_per_class = torch.bincount(labels, minlength=num_classes).float()
    # clamp avoids dividing by zero if a class had no images (not the case here, but safe)
    return correct_per_class / total_per_class.clamp(min=1)


def average_per_class_accuracy(logits, labels, num_classes):
    """Ave = (1/C) * sum_i T_i - every species counts equally, however many test images it has."""
    return per_class_accuracy(logits, labels, num_classes).mean().item()


def compute_metrics(logits, labels, num_classes):
    """All metrics in one dictionary, used for validation and test reporting."""
    return {
        "top1": top1_accuracy(logits, labels),
        "top5": topk_accuracy(logits, labels, k=5),
        "per_class_avg": average_per_class_accuracy(logits, labels, num_classes),
    }
