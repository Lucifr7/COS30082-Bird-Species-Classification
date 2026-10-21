"""
Phase 3 - Model architectures.

  * SimpleCNN   - a small CNN trained from scratch (experiment E1, the baseline)
  * build_model - creates either SimpleCNN or a pretrained CNN from timm
                  (ResNet-50, ConvNeXt-Tiny, ...) with a new 200-class output layer
"""

import timm
import torch
import torch.nn as nn

NUM_CLASSES = 200


# ---------------------------------------------------------------------------
# E1 baseline: a CNN built from scratch
# ---------------------------------------------------------------------------
def conv_block(in_channels, out_channels):
    """Two 3x3 convolutions followed by 2x2 max pooling.

    Conv2d       - slides learnable 3x3 filters over the image to detect local patterns
                   (edges first, then textures, then parts like beaks and wings in deeper blocks)
    BatchNorm2d  - re-centres and re-scales each channel over the batch; makes training faster
                   and more stable, and adds a little regularisation
    ReLU         - non-linearity max(0, x); without it stacked convolutions would collapse
                   into one linear operation
    MaxPool2d(2) - keeps the strongest response in each 2x2 window, halving height and width.
                   This makes features slightly position-invariant and lets later filters
                   cover a larger area of the original image
    padding=1 keeps the height and width unchanged through each convolution.
    bias=False because BatchNorm straight after already adds its own bias.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SimpleCNN(nn.Module):
    """Four conv blocks (32 -> 64 -> 128 -> 256 channels), global average pooling, linear classifier.

    Shape of the data as it flows through (for a 224x224 input):
        (3, 224, 224) -> block1 -> (32, 112, 112) -> block2 -> (64, 56, 56)
        -> block3 -> (128, 28, 28) -> block4 -> (256, 14, 14)
        -> global average pool -> (256,) -> dropout -> linear -> (200,) class scores
    """

    def __init__(self, num_classes=NUM_CLASSES, drop_rate=0.5):
        super().__init__()  # sets up nn.Module internals so PyTorch can track the layers
        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )
        # Average each of the 256 feature maps down to a single number. Much fewer
        # parameters than flattening 256x14x14 = 50,176 values into the linear layer,
        # which would overfit badly on this small dataset.
        self.pool = nn.AdaptiveAvgPool2d(1)
        # Dropout randomly zeroes 50% of the features during training so the classifier
        # can't rely on any single feature. It is switched off automatically in eval mode.
        self.dropout = nn.Dropout(drop_rate)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)  # (batch, 256, 1, 1) -> (batch, 256)
        x = self.dropout(x)
        return self.classifier(x)    # raw scores ("logits"); softmax is applied inside the loss


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def build_model(name, pretrained=True, freeze_backbone=False, drop_rate=0.0, num_classes=NUM_CLASSES):
    """Create a model by name.

    name="simple_cnn" builds the scratch baseline. Any other name is passed to timm,
    e.g. "resnet50.tv_in1k" (ResNet-50 with the original torchvision ImageNet weights) or
    "convnext_tiny.fb_in1k" (ConvNeXt-Tiny trained on ImageNet-1k by Facebook AI).

    num_classes=200 tells timm to throw away the 1000-class ImageNet output layer and put
    a new, randomly initialised 200-class layer in its place. Everything before that layer
    keeps its pretrained weights.
    """
    if name == "simple_cnn":
        return SimpleCNN(num_classes, drop_rate=drop_rate or 0.5)

    model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)

    if freeze_backbone:
        # Transfer learning with a frozen feature extractor (E2):
        # turn off gradients for every weight, then turn them back on for the new head only.
        for p in model.parameters():
            p.requires_grad = False
        for p in model.get_classifier().parameters():
            p.requires_grad = True
    return model


def set_train_mode(model):
    """Put the model in training mode, but keep frozen BatchNorm layers in eval mode.

    In training mode BatchNorm updates its running mean/variance from each batch. For a
    frozen backbone we want the ImageNet statistics left untouched, otherwise the frozen
    features would slowly drift even though no weights are being trained.
    """
    model.train()
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm) and not any(p.requires_grad for p in m.parameters()):
            m.eval()


def get_param_groups(model, lr_head, lr_backbone, weight_decay):
    """Split trainable parameters into two groups with different learning rates.

    The new head starts from random weights and needs a large learning rate to learn quickly.
    The pretrained backbone is already good; a small learning rate adjusts it gently instead
    of destroying what it learned on ImageNet ("catastrophic forgetting").
    For SimpleCNN everything is trained from scratch, so everything uses lr_head.
    """
    if isinstance(model, SimpleCNN):
        return [{"params": model.parameters(), "lr": lr_head, "weight_decay": weight_decay}]

    head_ids = {id(p) for p in model.get_classifier().parameters()}
    head, backbone = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue  # frozen weights are not given to the optimiser at all
        (head if id(p) in head_ids else backbone).append(p)

    groups = [{"params": head, "lr": lr_head, "weight_decay": weight_decay}]
    if backbone:
        groups.append({"params": backbone, "lr": lr_backbone, "weight_decay": weight_decay})
    return groups


def load_trained_model(checkpoint_path, device="cpu"):
    """Rebuild a model from a best.pth checkpoint saved by train.py.

    Returns (model, class_names, args). pretrained=False because the ImageNet weights are
    about to be overwritten by our own trained weights anyway - no need to download them.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    args = ckpt["args"]
    model = build_model(args["model"], pretrained=False, drop_rate=args["drop_rate"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt["class_names"], args


def get_cam_target_layer(model):
    """The layer Grad-CAM should look at: the last convolutional stage, just before pooling.

    It is the deepest layer that still has a spatial layout (e.g. 7x7 for a 224 input), so its
    activations combine high-level meaning ("this is a beak") with where in the image it is.
    """
    if isinstance(model, SimpleCNN):
        return model.features[-1]
    if hasattr(model, "stages"):   # ConvNeXt
        return model.stages[-1]
    if hasattr(model, "layer4"):   # ResNet
        return model.layer4[-1]
    raise ValueError(f"No Grad-CAM layer defined for {type(model).__name__}")


def count_parameters(model):
    """Return (total, trainable) parameter counts - useful for the report's model comparison."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
