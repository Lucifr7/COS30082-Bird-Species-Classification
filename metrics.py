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


def expected_calibration_error(logits, labels, n_bins=15):
    """ECE: how far the model's confidence is from its actual accuracy (0 = perfectly calibrated).

    Predictions are grouped into bins by confidence (0-6.7%, 6.7-13.3%, ...). In each bin we compare
    the average confidence with the fraction actually correct, and take the weighted average of the
    gaps. E.g. if images predicted with ~50% confidence are right 85% of the time, the model is
    under-confident by 35 points in that bin.
    """
    probs = torch.softmax(logits, dim=1)
    confidence, preds = probs.max(dim=1)
    correct = (preds == labels).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lo) & (confidence <= hi)
        if in_bin.any():
            gap = (confidence[in_bin].mean() - correct[in_bin].mean()).abs()
            ece += gap.item() * in_bin.float().mean().item()  # weight by fraction of images in the bin
    return ece


def fit_temperature(logits, labels):
    """Temperature scaling: find T > 0 so that softmax(logits / T) is calibrated.

    T is fitted on the validation set by minimising the cross-entropy (negative log-likelihood).
    Dividing all logits by the same T never changes the argmax, so accuracy is unchanged.
    We optimise log(T) rather than T so that T stays positive.
    """
    log_t = torch.zeros(1, requires_grad=True)  # start at T = exp(0) = 1, i.e. no change
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return log_t.exp().item()


def compute_metrics(logits, labels, num_classes):
    """All metrics in one dictionary, used for validation and test reporting."""
    return {
        "top1": top1_accuracy(logits, labels),
        "top5": topk_accuracy(logits, labels, k=5),
        "per_class_avg": average_per_class_accuracy(logits, labels, num_classes),
    }
