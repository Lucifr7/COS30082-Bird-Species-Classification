"""
Phase 1 - Exploratory data analysis of the CUB-200 dataset.

Answers three questions before any model is built:
  1. How many images per class? (decides the validation split and why per-class accuracy matters)
  2. How big are the images and are they all colour? (decides resize size and RGB conversion)
  3. What do the birds look like? (decides which augmentations make sense)
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

# ---------------------------------------------------------------------------
# Paths - kept in one place so they are easy to change
# ---------------------------------------------------------------------------
DATA_DIR = "data"
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)  # create figures/ if it does not exist yet


def load_annotations(split):
    """Read train.txt or test.txt into a DataFrame with columns 'file' and 'label'.

    Each line looks like "Black_footed_Albatross_0004_2731401028.jpg 0":
    a filename and an integer class label separated by one space, with no header row.
    """
    path = os.path.join(DATA_DIR, f"{split}.txt")
    return pd.read_csv(path, sep=" ", header=None, names=["file", "label"])


# ---------------------------------------------------------------------------
# Part A - load the annotations and check they parsed correctly
# ---------------------------------------------------------------------------
train_df = load_annotations("train")
test_df = load_annotations("test")

print("=== Part A: annotations ===")
print(train_df.head())  # first 5 rows as a sanity check
print(f"Train images: {len(train_df)}   Test images: {len(test_df)}")
print(f"Classes: {train_df['label'].nunique()}  (labels {train_df['label'].min()} to {train_df['label'].max()})")

# Class names are not given separately, but every filename starts with the species name,
# e.g. "Black_footed_Albatross_0004_2731401028.jpg" -> "Black footed Albatross".
# Dropping the last two underscore-separated parts (image number and Flickr id) leaves the name.
train_df["species"] = train_df["file"].apply(lambda f: " ".join(f.split("_")[:-2]))
class_names = train_df.drop_duplicates("label").sort_values("label")["species"].tolist()
print("First 5 class names:", class_names[:5])

# ---------------------------------------------------------------------------
# Part B - class balance
# ---------------------------------------------------------------------------
print("\n=== Part B: images per class ===")
train_counts = train_df["label"].value_counts().sort_index()  # index = label, value = count
test_counts = test_df["label"].value_counts().sort_index()

for name, counts in [("Train", train_counts), ("Test", test_counts)]:
    print(f"{name}: min={counts.min()}  max={counts.max()}  mean={counts.mean():.1f}")

# Bar chart of images per class - goes into the report to show the dataset is small and uneven
fig, ax = plt.subplots(figsize=(14, 4))
ax.bar(train_counts.index, train_counts.values, width=1.0, label="Train")
ax.bar(test_counts.index, test_counts.values, width=1.0, label="Test")
ax.set_xlabel("Class label")
ax.set_ylabel("Number of images")
ax.set_title("Images per class in CUB-200")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "class_distribution.png"), dpi=150)
plt.close(fig)  # free memory; we only want the saved file

# ---------------------------------------------------------------------------
# Part C - image sizes and colour modes (checks every training image)
# ---------------------------------------------------------------------------
print("\n=== Part C: image sizes and modes ===")
widths, heights, modes = [], [], {}
for filename in train_df["file"]:
    # 'with' closes the file afterwards; PIL reads only the header here, so this is fast
    with Image.open(os.path.join(DATA_DIR, "Train", filename)) as img:
        w, h = img.size  # PIL gives (width, height)
        widths.append(w)
        heights.append(h)
        modes[img.mode] = modes.get(img.mode, 0) + 1  # count how many of each mode

print(f"Width : min={min(widths)}  max={max(widths)}  mean={sum(widths) / len(widths):.0f}")
print(f"Height: min={min(heights)}  max={max(heights)}  mean={sum(heights) / len(heights):.0f}")
print("Colour modes found:", modes)  # anything other than 'RGB' must be converted before training

# ---------------------------------------------------------------------------
# Part D - grid of random sample images
# ---------------------------------------------------------------------------
samples = train_df.sample(16, random_state=42)  # fixed seed -> same 16 images every run

fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for ax, (_, row) in zip(axes.flat, samples.iterrows()):
    img = Image.open(os.path.join(DATA_DIR, "Train", row["file"])).convert("RGB")
    ax.imshow(img)
    ax.set_title(f"{row['label']}: {row['species']}", fontsize=9)
    ax.axis("off")  # hide tick marks; they mean nothing for photos
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "samples.png"), dpi=120)
plt.close(fig)

print(f"\nFigures saved to {FIG_DIR}/")
