"""
Phase 3 - Training script. One script runs every experiment; command-line flags pick the settings.

Examples (see README for the full list of experiments):
    python train.py --name e1_simple_cnn --model simple_cnn --no-pretrained --epochs 60
    python train.py --name e2_resnet50_frozen --model resnet50.tv_in1k --freeze --epochs 15
    python train.py --name e3_resnet50_finetune --model resnet50.tv_in1k

Each run writes to runs/<name>/:
    best.pth       weights of the epoch with the highest validation accuracy
    history.csv    loss / accuracy / learning rate for every epoch
    curves.png     training vs validation curves (shows overfitting at a glance)
    results.json   final train / val / test metrics of the best epoch
"""

import argparse
import json
import math
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

from dataset import DATA_DIR, BirdDataset, get_class_names, get_dataloaders, get_transforms, split_train_val
from metrics import compute_metrics
from models import NUM_CLASSES, build_model, count_parameters, get_param_groups, set_train_mode


# ---------------------------------------------------------------------------
# Command-line arguments - every hyperparameter can be changed without editing code
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train a bird species classifier on CUB-200")
    p.add_argument("--name", required=True, help="run name; outputs go to runs/<name>/")
    p.add_argument("--model", default="resnet50.tv_in1k", help="'simple_cnn' or any timm model name")
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True,
                   help="start from ImageNet weights (--no-pretrained to train from scratch)")
    p.add_argument("--freeze", action="store_true", help="freeze the backbone, train only the new head")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--lr", type=float, default=1e-3, help="learning rate of the new head (and of SimpleCNN)")
    p.add_argument("--lr-backbone", type=float, default=1e-4, help="learning rate of the pretrained backbone")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--mixup-prob", type=float, default=0.0, help="probability of applying Mixup/CutMix to a batch")
    p.add_argument("--drop-rate", type=float, default=0.0, help="dropout before the classifier")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed):
    """Seed every random number generator so a run can be repeated with the same result."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Learning-rate schedule: linear warmup, then cosine decay
# ---------------------------------------------------------------------------
def make_scheduler(optimizer, warmup_steps, total_steps):
    """Return a scheduler that multiplies every group's learning rate by a factor in [0, 1].

    Warmup: the factor rises linearly from ~0 to 1 over the first warmup_steps batches.
            At the start the new head is random and produces large, noisy gradients;
            small steps stop those from damaging the pretrained backbone.
    Cosine: the factor then follows half a cosine wave from 1 down to 0. Big steps early
            explore quickly; tiny steps at the end let the weights settle into a minimum.
    """
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


# ---------------------------------------------------------------------------
# One epoch of training
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, mixup, device):
    set_train_mode(model)  # enables dropout and BatchNorm batch statistics
    total_loss, correct, seen = 0.0, 0, 0

    for images, labels in loader:
        # non_blocking lets the copy to the GPU overlap with computation (works with pin_memory)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Mixup/CutMix: blend pairs of images in the batch and turn the hard labels into
        # mixed soft labels, e.g. 0.7 cardinal + 0.3 blue jay. targets is then (B, 200).
        targets = mixup(images, labels) if mixup else labels
        if mixup:
            images, targets = targets

        # Mixed precision: run the forward pass in float16 where it is safe.
        # Roughly halves memory and doubles speed on the RTX 3060's tensor cores.
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(images)               # (B, 200) raw class scores
            loss = criterion(logits, targets)    # one number: how wrong the batch was

        optimizer.zero_grad(set_to_none=True)    # clear gradients left over from the previous batch
        # float16 gradients can underflow to zero; the scaler multiplies the loss by a large
        # factor before backward(), then divides the gradients back before the update.
        scaler.scale(loss).backward()            # backpropagation: d(loss)/d(weight) for every weight
        scaler.step(optimizer)                   # weight update: w <- w - lr * (AdamW-adjusted gradient)
        scaler.update()
        scheduler.step()                         # move the learning rate along the warmup/cosine curve

        total_loss += loss.item() * images.size(0)
        # Accuracy against the original labels. With Mixup the images are blends, so this is
        # only a rough indicator; the clean train accuracy is measured at the end of the run.
        correct += (logits.argmax(1) == labels).sum().item()
        seen += images.size(0)

    return total_loss / seen, correct / seen


# ---------------------------------------------------------------------------
# Evaluation (validation / test)
# ---------------------------------------------------------------------------
@torch.no_grad()  # no gradients needed: saves memory and time
def evaluate(model, loader, device):
    """Run the model over a whole loader and return (loss, metrics dict, all logits, all labels)."""
    model.eval()  # disables dropout; BatchNorm uses its stored running statistics
    criterion = nn.CrossEntropyLoss()  # plain cross-entropy so losses are comparable between runs
    all_logits, all_labels = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(images)
        all_logits.append(logits.float().cpu())
        all_labels.append(labels)

    logits = torch.cat(all_logits)  # (N, 200) for the whole set
    labels = torch.cat(all_labels)  # (N,)
    loss = criterion(logits, labels).item()
    return loss, compute_metrics(logits, labels, NUM_CLASSES), logits, labels


def plot_curves(history, path):
    """Two panels: loss and accuracy, training vs validation. The gap between the lines is overfitting."""
    df = pd.DataFrame(history)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(df["epoch"], df["train_loss"], label="train")
    ax1.plot(df["epoch"], df["val_loss"], label="validation")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax2.plot(df["epoch"], df["train_acc"], label="train")
    ax2.plot(df["epoch"], df["val_top1"], label="validation")
    ax2.set_title("Top-1 accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)
    # Let cuDNN benchmark and pick the fastest convolution algorithm for our input size
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")

    out_dir = os.path.join("runs", args.name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)  # record exactly which settings produced this run

    # ---- data ----
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size, img_size=args.img_size, num_workers=args.num_workers
    )

    # ---- model ----
    model = build_model(args.model, pretrained=args.pretrained, freeze_backbone=args.freeze,
                        drop_rate=args.drop_rate).to(device)
    total, trainable = count_parameters(model)
    print(f"Model {args.model}: {total / 1e6:.1f}M parameters, {trainable / 1e6:.2f}M trainable")

    # ---- loss ----
    mixup = None
    if args.mixup_prob > 0:
        # timm's Mixup randomly picks Mixup (alpha 0.8) or CutMix (alpha 1.0) for each batch,
        # and also applies label smoothing to the mixed targets
        mixup = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, prob=args.mixup_prob, switch_prob=0.5,
                      label_smoothing=args.label_smoothing, num_classes=NUM_CLASSES)
        # Targets are soft probability vectors, so use the soft-target version of cross-entropy
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # ---- optimiser, schedule, mixed precision ----
    optimizer = torch.optim.AdamW(get_param_groups(model, args.lr, args.lr_backbone, args.weight_decay))
    steps_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, args.warmup_epochs * steps_per_epoch, args.epochs * steps_per_epoch)
    scaler = torch.amp.GradScaler("cuda")

    # ---- training loop ----
    history, best_val = [], -1.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer,
                                                scheduler, scaler, mixup, device)
        val_loss, val_m, _, _ = evaluate(model, val_loader, device)

        row = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
               "val_loss": val_loss, "val_top1": val_m["top1"], "val_top5": val_m["top5"],
               "val_per_class": val_m["per_class_avg"], "lr_head": optimizer.param_groups[0]["lr"]}
        history.append(row)

        # Keep the weights from the epoch with the best validation accuracy. If the model
        # starts overfitting later, those later epochs simply never replace this checkpoint.
        is_best = val_m["top1"] > best_val
        if is_best:
            best_val = val_m["top1"]
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch,
                        "class_names": get_class_names()}, os.path.join(out_dir, "best.pth"))

        print(f"Epoch {epoch:3d}/{args.epochs} | train loss {train_loss:.3f} acc {train_acc:.3f} | "
              f"val loss {val_loss:.3f} top1 {val_m['top1']:.3f} per-class {val_m['per_class_avg']:.3f} | "
              f"{time.time() - start:.0f}s{'  *best*' if is_best else ''}")

        pd.DataFrame(history).to_csv(os.path.join(out_dir, "history.csv"), index=False)
        plot_curves(history, os.path.join(out_dir, "curves.png"))

    # ---- final evaluation of the best checkpoint ----
    ckpt = torch.load(os.path.join(out_dir, "best.pth"), map_location=device)
    model.load_state_dict(ckpt["model"])

    # Clean train accuracy: training images with the eval transform (no augmentation),
    # so it is directly comparable with validation accuracy. The difference is the overfitting gap.
    train_df, _ = split_train_val()
    _, eval_tf = get_transforms(args.img_size)
    clean_train_loader = torch.utils.data.DataLoader(
        BirdDataset(train_df, os.path.join(DATA_DIR, "Train"), eval_tf),
        batch_size=args.batch_size, num_workers=args.num_workers)

    _, train_m, _, _ = evaluate(model, clean_train_loader, device)
    _, val_m, _, _ = evaluate(model, val_loader, device)
    _, test_m, _, _ = evaluate(model, test_loader, device)

    results = {"name": args.name, "model": args.model, "best_epoch": ckpt["epoch"],
               "params_total": total, "params_trainable": trainable,
               "train": train_m, "val": val_m, "test": test_m}
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBest epoch {ckpt['epoch']}")
    for split, m in [("train", train_m), ("val", val_m), ("test", test_m)]:
        print(f"  {split:5s}: top1 {m['top1']:.4f}  per-class {m['per_class_avg']:.4f}  top5 {m['top5']:.4f}")
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
