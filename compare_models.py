"""
McNemar's test: is one model significantly better than another on the same test set?

    python compare_models.py --a runs/e5_convnext_reg --b runs/e6_convnext_384

Accuracy alone can mislead on a small test set (1,204 images): a 2-point difference may be chance.
McNemar's test compares two classifiers image by image and looks only at the images where they
DISAGREE:
    b = images model A got right and model B got wrong
    c = images model A got wrong and model B got right
If the models were equally good, b and c would be about equal. The statistic
    chi2 = (b - c)^2 / (b + c)
follows a chi-squared distribution with 1 degree of freedom, which gives the p-value.
p < 0.05 means the difference is statistically significant.
"""

import argparse
import os

import torch
from scipy.stats import chi2

from dataset import get_dataloaders
from models import load_trained_model
from train import evaluate


def test_predictions(run_dir, device):
    """Load a run's best checkpoint and return (predictions, labels) on the test set."""
    model, _, args = load_trained_model(os.path.join(run_dir, "best.pth"), device)
    # each model is evaluated at its own input size (224 or 384)
    _, _, test_loader = get_dataloaders(batch_size=32, img_size=args["img_size"])
    _, _, logits, labels = evaluate(model, test_loader, device)
    return logits.argmax(1), labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", default="runs/e5_convnext_reg", help="baseline model run folder")
    p.add_argument("--b", default="runs/e6_convnext_384", help="model being compared")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preds_a, labels = test_predictions(args.a, device)
    preds_b, _ = test_predictions(args.b, device)  # same test set, same order (no shuffling)
    right_a, right_b = preds_a == labels, preds_b == labels

    # the 2x2 table of agreements and disagreements
    both = (right_a & right_b).sum().item()
    only_a = (right_a & ~right_b).sum().item()  # b in the formula
    only_b = (~right_a & right_b).sum().item()  # c in the formula
    neither = (~right_a & ~right_b).sum().item()

    stat = (only_a - only_b) ** 2 / (only_a + only_b)
    p_value = chi2.sf(stat, df=1)  # sf = 1 - CDF: probability of a statistic at least this large

    print(f"A = {args.a}: {right_a.float().mean():.4f} test Top-1")
    print(f"B = {args.b}: {right_b.float().mean():.4f} test Top-1")
    print(f"both right {both} | only A right {only_a} | only B right {only_b} | both wrong {neither}")
    print(f"McNemar chi2 = {stat:.2f}, p = {p_value:.4f} -> "
          f"{'significant' if p_value < 0.05 else 'not significant'} at 0.05")


if __name__ == "__main__":
    main()
