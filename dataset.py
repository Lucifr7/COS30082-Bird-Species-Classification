"""
Phase 2 - Data pipeline for CUB-200.

Provides everything the training script needs to get batches of images:
  * a stratified train/validation split of the official training set
  * a PyTorch Dataset that turns (filename, label) rows into image tensors
  * train-time (augmented) and eval-time (deterministic) transforms
  * DataLoaders that batch and shuffle the data and load it in parallel

Run this file directly to sanity-check the pipeline and save a preview of augmented images.
"""

import os

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

DATA_DIR = "data"
SEED = 42

# Mean and standard deviation of each RGB channel over ImageNet.
# Pretrained models were trained on images normalised with these values,
# so our images must be normalised the same way for the pretrained weights to make sense.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Annotations and the train/validation split
# ---------------------------------------------------------------------------
def load_annotations(split):
    """Read train.txt or test.txt into a DataFrame with columns 'file' and 'label'."""
    path = os.path.join(DATA_DIR, f"{split}.txt")
    return pd.read_csv(path, sep=" ", header=None, names=["file", "label"])


def get_class_names():
    """Return a list where index i is the species name of class i (taken from the filenames)."""
    df = load_annotations("train")
    df["species"] = df["file"].apply(lambda f: " ".join(f.split("_")[:-2]))
    return df.drop_duplicates("label").sort_values("label")["species"].tolist()


def split_train_val(val_fraction=0.2, seed=SEED):
    """Split the official training set into our own training and validation sets.

    The test set is only used once, for the final reported numbers. Hyperparameters and
    early stopping are decided on this validation set, so the test score stays an honest
    estimate of performance on unseen data.

    stratify=label keeps the same fraction of every class in both parts, so each of the
    200 species keeps about 80% of its images for training and about 20% (3-6 images)
    for validation - similar to the 4-8 per class in the test set.
    The fixed seed means the split is identical on every run.
    """
    df = load_annotations("train")
    train_df, val_df = train_test_split(
        df, test_size=val_fraction, stratify=df["label"], random_state=seed
    )
    # reset_index renumbers rows 0..N-1 so the Dataset can index them by position
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class BirdDataset(Dataset):
    """Maps an index to (image_tensor, label).

    A PyTorch Dataset only needs two methods: __len__ (how many samples) and
    __getitem__ (fetch sample i). The DataLoader calls __getitem__ for each index
    in a batch, so images are read from disk lazily, one at a time, instead of
    loading all ~6000 images into memory at once.
    """

    def __init__(self, df, img_dir, transform=None):
        self.files = df["file"].tolist()
        self.labels = df["label"].tolist()
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, self.files[idx])
        # convert("RGB") guarantees 3 channels even for greyscale or RGBA images
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)  # PIL image -> augmented, normalised tensor
        return image, self.labels[idx]


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def get_transforms(img_size=224):
    """Return (train_transform, eval_transform).

    Training uses random augmentation: every epoch the model sees a slightly different
    version of each image, which acts like a larger dataset and reduces overfitting.
    Evaluation must be deterministic, so the same image always gives the same prediction.
    """
    train_transform = transforms.Compose([
        # Crop a random region covering 50-100% of the image, then resize to img_size.
        # Teaches the model to recognise birds at different scales and positions.
        transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
        # Mirror half of the images. A bird facing left is still the same species.
        transforms.RandomHorizontalFlip(p=0.5),
        # Small random changes in brightness/contrast/saturation simulate different lighting.
        # Hue is left alone because colour is often what separates two species.
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        # PIL image (H x W x 3, values 0-255) -> float tensor (3 x H x W, values 0-1)
        transforms.ToTensor(),
        # Subtract the ImageNet mean and divide by the std, per channel
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        # Resize the short side to a bit more than img_size (256 for 224), then take the centre.
        # This is the standard ImageNet evaluation recipe and keeps the aspect ratio intact.
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_transform, eval_transform


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------
def get_dataloaders(batch_size=32, img_size=224, num_workers=4, val_fraction=0.2):
    """Build the train, validation and test DataLoaders."""
    train_tf, eval_tf = get_transforms(img_size)
    train_df, val_df = split_train_val(val_fraction)
    test_df = load_annotations("test")

    train_dir = os.path.join(DATA_DIR, "Train")
    test_dir = os.path.join(DATA_DIR, "Test")

    # Validation images come from the Train folder but use the eval transform (no augmentation)
    train_ds = BirdDataset(train_df, train_dir, train_tf)
    val_ds = BirdDataset(val_df, train_dir, eval_tf)
    test_ds = BirdDataset(test_df, test_dir, eval_tf)

    loader_args = dict(
        batch_size=batch_size,
        num_workers=num_workers,           # parallel processes that load/augment images while the GPU trains
        pin_memory=True,                   # page-locked memory makes CPU -> GPU copies faster
        persistent_workers=num_workers > 0,  # keep workers alive between epochs (worker start-up is slow on Windows)
    )
    # shuffle=True on training data so batches differ every epoch and the model can't learn the order.
    # drop_last=True discards a final tiny batch, which would give a noisy gradient step.
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_args)
    # No shuffling for evaluation - order doesn't matter and fixed order makes results reproducible
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_args)
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Sanity check - run "python dataset.py"
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # The guard above is required on Windows: each DataLoader worker re-imports this file,
    # and without the guard every worker would start building DataLoaders again.
    import matplotlib.pyplot as plt

    train_loader, val_loader, test_loader = get_dataloaders()
    print(f"Train: {len(train_loader.dataset)} images, {len(train_loader)} batches")
    print(f"Val  : {len(val_loader.dataset)} images, {len(val_loader)} batches")
    print(f"Test : {len(test_loader.dataset)} images, {len(test_loader)} batches")

    # Check the stratification worked: every class should appear in both parts
    train_df, val_df = split_train_val()
    val_counts = val_df["label"].value_counts()
    print(f"Classes in train split: {train_df['label'].nunique()}, in val split: {val_df['label'].nunique()}")
    print(f"Val images per class: min={val_counts.min()}  max={val_counts.max()}")
    overlap = set(train_df["file"]) & set(val_df["file"])
    print(f"Images in both train and val (must be 0): {len(overlap)}")

    # Fetch one batch and inspect what the model will actually receive
    images, labels = next(iter(train_loader))
    print(f"Batch images: shape={tuple(images.shape)} dtype={images.dtype} "
          f"min={images.min():.2f} max={images.max():.2f}")
    print(f"Batch labels: shape={tuple(labels.shape)} first 8={labels[:8].tolist()}")

    # Undo the normalisation so the augmented images can be displayed as normal photos
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    class_names = get_class_names()

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for ax, img, label in zip(axes.flat, images, labels):
        img = (img * std + mean).clamp(0, 1)   # back to the 0-1 range
        ax.imshow(img.permute(1, 2, 0))       # (3, H, W) -> (H, W, 3), which matplotlib expects
        ax.set_title(class_names[label], fontsize=9)
        ax.axis("off")
    fig.suptitle("Augmented training batch")
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/augmented_batch.png", dpi=120)
    print("Saved figures/augmented_batch.png")
