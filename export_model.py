"""
Phase 6 - Package the best model and its assets into deploy/ for the Hugging Face Space.

    python export_model.py --run runs/e6_convnext_384

Creates:
    deploy/model.pth           trained weights (float16 to halve the file size) + metadata
                               + calibration temperature
    deploy/reference/NNN.jpg   one small training photo per species, shown next to the prediction
    deploy/examples/*.jpg      test photos for the app's clickable examples
    deploy/results_table.csv   + model_comparison.png + calibration.png, shown on the app's "About" tab
(run evaluate.py first - it creates the figures copied here)

The deploy/ folder is self-contained: it is exactly what gets uploaded to the Space.
"""

import argparse
import json
import os
import shutil

import torch
from PIL import Image

from dataset import DATA_DIR, get_dataloaders, load_annotations
from metrics import fit_temperature
from models import load_trained_model
from train import evaluate

DEPLOY_DIR = "deploy"

# Test images used as clickable examples: a spread of families, plus two hard cases
# (look-alike terns and sparrows) that are useful for discussing limitations in the video.
EXAMPLE_SPECIES = ["Cardinal", "Blue Jay", "Brown Pelican", "Rhinoceros Auklet", "Magnolia Warbler",
                   "Ruby throated Hummingbird", "Painted Bunting", "Belted Kingfisher",
                   "Common Tern", "Field Sparrow"]
HARD_SPECIES = {"Common Tern", "Field Sparrow"}  # for these, deliberately pick an image the model gets wrong


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/e6_convnext_384")
    args = p.parse_args()
    os.makedirs(os.path.join(DEPLOY_DIR, "reference"), exist_ok=True)
    os.makedirs(os.path.join(DEPLOY_DIR, "examples"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, run_args = load_trained_model(os.path.join(args.run, "best.pth"), device)
    with open(os.path.join(args.run, "results.json")) as f:
        results = json.load(f)
    _, val_loader, test_loader = get_dataloaders(batch_size=32, img_size=run_args["img_size"])

    # ---- 1. temperature for calibrated confidences, fitted on the validation set ----
    # Label smoothing and Mixup make the model under-confident; dividing the logits by T < 1
    # sharpens the probabilities so the confidence shown in the app matches real accuracy.
    _, _, val_logits, val_labels = evaluate(model, val_loader, device)
    temperature = fit_temperature(val_logits, val_labels)
    print(f"Temperature T = {temperature:.3f}")

    # ---- 2. slim checkpoint ----
    # float16 halves the file (~110 MB -> ~56 MB) for a faster upload and Space start-up.
    # The app converts back to float32 when loading, so predictions are practically identical.
    state = {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}
    torch.save({
        "model": state,
        "model_name": run_args["model"],
        "img_size": run_args["img_size"],
        "class_names": class_names,
        "temperature": temperature,
        "test_metrics": results["test"],
    }, os.path.join(DEPLOY_DIR, "model.pth"))
    print(f"Saved model.pth ({os.path.getsize(os.path.join(DEPLOY_DIR, 'model.pth')) / 1e6:.0f} MB)")

    # ---- 3. one reference photo per species (first training image, shrunk to save space) ----
    train_df = load_annotations("train")
    for label, group in train_df.groupby("label"):
        img = Image.open(os.path.join(DATA_DIR, "Train", group["file"].iloc[0])).convert("RGB")
        img.thumbnail((320, 320))  # resize in place, keeping the aspect ratio
        img.save(os.path.join(DEPLOY_DIR, "reference", f"{label:03d}.jpg"), quality=85)
    print(f"Saved {train_df['label'].nunique()} reference images")

    # ---- 4. example images: a correctly classified test image of each species,
    #         except for HARD_SPECIES where a misclassified one is chosen to show a limitation ----
    _, _, logits, labels = evaluate(model, test_loader, device)
    preds = logits.argmax(1)
    test_df = load_annotations("test")
    for species in EXAMPLE_SPECIES:
        label = class_names.index(species)
        candidates = [i for i in range(len(test_df)) if labels[i] == label]
        wanted_correct = species not in HARD_SPECIES
        matching = [i for i in candidates if (preds[i] == label) == wanted_correct]
        idx = (matching or candidates)[0]  # fall back to any image if none match
        print(f"  {species}: test image {test_df['file'][idx]}, predicted {class_names[preds[idx]]}")
        dst = os.path.join(DEPLOY_DIR, "examples", species.replace(" ", "_") + ".jpg")
        shutil.copy(os.path.join(DATA_DIR, "Test", test_df["file"][idx]), dst)
    print(f"Saved {len(EXAMPLE_SPECIES)} example images")

    # ---- 5. results for the About tab ----
    for name in ["results_table.csv", "model_comparison.png", "calibration.png"]:
        shutil.copy(os.path.join("figures", name), os.path.join(DEPLOY_DIR, name))
    print("Copied results table and comparison chart")


if __name__ == "__main__":
    main()
