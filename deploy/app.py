"""
Bird species classifier - Gradio web app for Hugging Face Spaces.

Upload a bird photo and the app shows:
  * the 5 most likely species with confidence bars
  * a Grad-CAM heatmap of the image regions behind the prediction (for any of the top 5)
  * a reference photo of the predicted species from the training set, to compare by eye

This file is self-contained (it does not import the training code), so the deploy/ folder
can be uploaded to a Space on its own. Run locally with:  python app.py
"""

import os

import gradio as gr
import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms

# ZeroGPU (the free Hugging Face Spaces hardware) lends the Space a GPU only while a function decorated
# with @spaces.GPU runs. 'duration' is the maximum seconds per call; a short value gets queued sooner.
# On a normal computer the 'spaces' package is absent, so the decorator falls back to doing nothing.
try:
    import spaces
    gpu = spaces.GPU(duration=20)
except ImportError:
    def gpu(fn):
        return fn

HERE = os.path.dirname(os.path.abspath(__file__))  # folder of this file, wherever it is run from
LOW_CONFIDENCE = 0.5  # below this top-1 probability the app shows a warning
# Below this the input is probably not one of the 200 species (or not a bird at all).
# Measured on the test set: no real test bird had calibrated confidence under 10%, and only 0.2%
# were under 15% (all of them misclassified anyway), so this threshold rarely rejects a real bird.
NOT_A_BIRD = 0.15

# ---------------------------------------------------------------------------
# Load the model once, when the app starts
# ---------------------------------------------------------------------------
# On ZeroGPU, moving the model to "cuda" at start-up is allowed: the move happens when a GPU is lent
ON_ZERO_GPU = os.environ.get("SPACES_ZERO_GPU") is not None
device = torch.device("cuda" if torch.cuda.is_available() or ON_ZERO_GPU else "cpu")
ckpt = torch.load(os.path.join(HERE, "model.pth"), map_location="cpu")
CLASS_NAMES = ckpt["class_names"]
IMG_SIZE = ckpt["img_size"]
# Temperature scaling (fitted on the validation set): label smoothing and Mixup make the model
# under-confident, so logits are divided by T < 1 to make the displayed confidence match accuracy
TEMPERATURE = ckpt.get("temperature", 1.0)

# Same architecture as in training, with a 200-class head; pretrained=False because
# our own trained weights are loaded straight after
model = timm.create_model(ckpt["model_name"], pretrained=False, num_classes=len(CLASS_NAMES))
# The weights were stored as float16 to shrink the file; .float() turns them back into float32
model.load_state_dict({k: v.float() for k, v in ckpt["model"].items()})
model.to(device).eval()

# Grad-CAM attached to the last ConvNeXt stage (the deepest layer that still has a spatial map)
cam = GradCAM(model=model, target_layers=[model.stages[-1]])

# Exactly the evaluation preprocessing used when the model was tested:
# resize the short side, centre-crop a square, convert to tensor, ImageNet normalisation
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
crop = transforms.Compose([transforms.Resize(int(IMG_SIZE * 256 / 224)), transforms.CenterCrop(IMG_SIZE)])
to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


# ---------------------------------------------------------------------------
# Core functions called by the interface
# ---------------------------------------------------------------------------
def _explain(image, class_name):
    """Grad-CAM overlay for one class, plus a reference training photo of that class."""
    if image is None or not class_name:
        return None, None
    label = CLASS_NAMES.index(class_name)
    cropped = crop(image.convert("RGB"))                 # the exact square the model looks at
    x = to_tensor(cropped).unsqueeze(0).to(device)       # (1, 3, H, W) batch of one
    heatmap = cam(input_tensor=x, targets=[ClassifierOutputTarget(label)])[0]
    rgb = np.asarray(cropped, dtype=np.float32) / 255.0  # show_cam_on_image expects 0-1 floats
    overlay = show_cam_on_image(rgb, heatmap, use_rgb=True, image_weight=0.55)
    reference = Image.open(os.path.join(HERE, "reference", f"{label:03d}.jpg"))
    return overlay, reference


# Two entry points that need the GPU. predict() calls the undecorated _explain() so that it does
# not request a second GPU slot while it already holds one.
@gpu
def explain(image, class_name):
    return _explain(image, class_name)


@gpu
def predict(image):
    """Classify an uploaded image. Returns values for every output component, in order."""
    if image is None:
        return None, "", gr.update(choices=[], value=None), None, None

    x = to_tensor(crop(image.convert("RGB"))).unsqueeze(0).to(device)
    with torch.no_grad():
        # logits -> calibrated probabilities that sum to 1
        probs = torch.softmax(model(x)[0] / TEMPERATURE, dim=0)
    top_p, top_i = probs.topk(5)
    top5 = {CLASS_NAMES[i]: float(p) for p, i in zip(top_p, top_i)}  # gr.Label wants {name: prob}
    best_name, best_p = CLASS_NAMES[top_i[0]], float(top_p[0])

    if best_p < NOT_A_BIRD:
        note = (f"🚫 **This doesn't look like any of the 200 species I know** (best guess only {best_p:.0%}). "
                "It may not be a bird, or the bird is too small or hidden. The guesses below are not reliable.")
    elif best_p < LOW_CONFIDENCE:
        note = (f"⚠️ **Low confidence ({best_p:.0%}).** Possibly **{best_name}**, but the model is unsure - "
                "the photo may be unclear, show several birds, or the species may be easy to confuse "
                "with a close relative. Compare with the reference photo.")
    else:
        note = f"✅ **{best_name}** with {best_p:.0%} confidence."

    overlay, reference = _explain(image, best_name)
    # The dropdown lets the user ask "why not the 2nd guess?" and see that class's heatmap instead
    dropdown = gr.update(choices=list(top5), value=best_name)
    return top5, note, dropdown, overlay, reference


# ---------------------------------------------------------------------------
# About tab content
# ---------------------------------------------------------------------------
def results_markdown():
    """Turn results_table.csv into a markdown table for the About tab."""
    df = pd.read_csv(os.path.join(HERE, "results_table.csv"))
    lines = ["| Experiment | Params (M) | Val Top-1 | Test Top-1 | Test avg per-class | Test Top-5 |",
             "|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(f"| {r['experiment']} | {r['params_M']:.1f} | {r['val_top1']:.1%} | {r['test_top1']:.1%} "
                     f"| {r['test_per_class']:.1%} | {r['test_top5']:.1%} |")
    return "\n".join(lines)


m = ckpt["test_metrics"]
ABOUT = f"""
### Model
**ConvNeXt-Tiny** (a modern convolutional neural network, 28M parameters) pretrained on ImageNet-1k and
fine-tuned on **CUB-200** (200 North American bird species, 4,829 training images - about 24 per species)
at **{IMG_SIZE}x{IMG_SIZE}** pixels.

**Test set (1,204 images):** Top-1 accuracy **{m['top1']:.1%}** · average per-class accuracy
**{m['per_class_avg']:.1%}** · Top-5 accuracy **{m['top5']:.1%}**

### Training recipe
- Cross-entropy loss with label smoothing 0.1, Mixup/CutMix on half of the batches, dropout 0.3
- AdamW optimiser, learning rate 1e-3 for the new classifier head and 1e-4 for the pretrained backbone,
  weight decay 0.05, 2-epoch warmup then cosine decay, mixed-precision training
- Augmentation: random resized crop, horizontal flip, colour jitter (no hue change - colour identifies species)
- 80/20 stratified train/validation split; the checkpoint with the best validation accuracy was kept

### All experiments
{results_markdown()}

### Calibrated confidence
Label smoothing and Mixup improve accuracy but make the network under-confident (it is right more often
than its probabilities claim). The app divides the scores by a temperature **T = {TEMPERATURE:.2f}**, fitted on
the validation set, so a displayed confidence of 80% means the model is right about 80% of the time.
This does not change which species is predicted.

### How to read the heatmap
Grad-CAM highlights the regions that increased the score of the selected species: **red = most
important**, blue = ignored. A good prediction focuses on the bird's head, bill, breast and wings -
the same field marks birdwatchers use.

### Limitations
Only the 200 species in CUB-200 are known; the model always picks one of them, so a confidence below
{NOT_A_BIRD:.0%} is flagged as "not one of the 200 species" (no real test bird scored below 10%). The image is centre-cropped to a square, so a bird near the
edge can be cut off, and very small or distant birds are hard. Most remaining mistakes are between
closely related species (terns, sparrows, vireos, flycatchers).
"""

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
HARD_EXAMPLES = ["Common_Tern.jpg", "Field_Sparrow.jpg"]  # images the model gets wrong, shown last
example_files = sorted(f for f in os.listdir(os.path.join(HERE, "examples")) if f not in HARD_EXAMPLES)
examples = [os.path.join(HERE, "examples", f) for f in example_files + HARD_EXAMPLES]

with gr.Blocks(title="Bird Species Classifier") as demo:
    gr.Markdown("# 🐦 Bird Species Classifier\n"
                "Identify **200 bird species** from a photo, and see *where* the network looked to decide.")

    with gr.Tab("Classify"):
        with gr.Row():
            # ---- left column: input ----
            with gr.Column(scale=1):
                image_in = gr.Image(type="pil", label="Bird photo", sources=["upload", "webcam", "clipboard"],
                                    height=340)
                with gr.Row():
                    clear_btn = gr.ClearButton(value="Clear")
                    submit_btn = gr.Button("Identify bird", variant="primary")
                gr.Examples(examples=examples, inputs=image_in, label="Try an example (the last two are hard cases)",
                            examples_per_page=10)

            # ---- right column: results ----
            with gr.Column(scale=1):
                note_out = gr.Markdown()
                label_out = gr.Label(num_top_classes=5, label="Top-5 predictions")
                class_pick = gr.Dropdown(label="Explain the prediction for", choices=[], interactive=True)
                with gr.Row():
                    cam_out = gr.Image(label="Where it looked", height=260, interactive=False)
                    ref_out = gr.Image(label="Reference photo", height=260, interactive=False)
                gr.Markdown("<small>**Heatmap (Grad-CAM):** red = regions that most increased the score of the "
                            "selected species, blue = ignored. **Reference photo:** a training image of the "
                            "selected species, to compare by eye.</small>")

    with gr.Tab("About the model"):
        gr.Markdown(ABOUT)
        with gr.Row():
            gr.Image(os.path.join(HERE, "model_comparison.png"), label="Test accuracy of every experiment",
                     interactive=False)
            gr.Image(os.path.join(HERE, "calibration.png"), label="Calibration before / after temperature scaling",
                     interactive=False)

    # ---- events ----
    outputs = [label_out, note_out, class_pick, cam_out, ref_out]
    # 'change' fires on upload, webcam capture, example click and clear, so results always match
    # the current image (predict() returns empty outputs when the image is cleared)
    image_in.change(predict, inputs=image_in, outputs=outputs)
    submit_btn.click(predict, inputs=image_in, outputs=outputs)
    # choosing another species in the dropdown only recomputes the heatmap and reference photo
    class_pick.input(explain, inputs=[image_in, class_pick], outputs=[cam_out, ref_out])
    clear_btn.add([image_in, label_out, note_out, cam_out, ref_out])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="emerald"))
