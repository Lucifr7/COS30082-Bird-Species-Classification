# Bird Species Classification (CUB-200) — COS30082 Assignment 1

**Nigel Zi Jun LING · 102779140**

Classifies 200 bird species from the CUB-200 dataset with convolutional neural networks.
Six experiments go from a CNN trained from scratch to a fine-tuned ConvNeXt-Tiny at 384×384.

- **Best model (E6):** 88.7% test Top-1 accuracy, 88.4% average per-class accuracy, 98.0% Top-5
- **Web app:** https://huggingface.co/spaces/Nigel77/bird-species-classifier

## Files

| File | Purpose |
|---|---|
| `explore.py` | Dataset statistics and figures (class balance, image sizes, sample grid) |
| `dataset.py` | Stratified 80/20 train/validation split, `Dataset` class, augmentation, DataLoaders |
| `models.py` | Scratch CNN, pretrained model factory (timm), freezing, two learning-rate groups, checkpoint loading |
| `metrics.py` | Top-1, Top-5, average per-class accuracy, calibration error (ECE), temperature scaling |
| `train.py` | One training script for every experiment (loss, AdamW, warmup + cosine schedule, mixed precision, best-checkpoint selection) |
| `evaluate.py` | Results table and chart for all runs; per-class accuracy, confusion analysis, calibration and Grad-CAM for one model |
| `compare_models.py` | McNemar's test between two models on the test set |
| `export_model.py` | Packages the best model and app assets into `deploy/` |
| `deploy/app.py` | Gradio web app (top-5 predictions, Grad-CAM, reference photo, calibrated confidence) |
| `runs/<experiment>/` | Per-experiment settings (`args.json`), per-epoch history, learning curves and final metrics |
| `figures/` | Figures and tables used in the report |

Not included because of size: the dataset (`data/`) and trained weights (`*.pth`).
The deployed E6 weights are in the Hugging Face Space.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Place the dataset as `data/Train/`, `data/Test/`, `data/train.txt` and `data/test.txt`.

## Reproducing the experiments

```
python train.py --name e1_simple_cnn        --model simple_cnn --no-pretrained --epochs 60 --warmup-epochs 3
python train.py --name e2_resnet50_frozen   --model resnet50.tv_in1k --freeze --epochs 20
python train.py --name e3_resnet50_finetune --model resnet50.tv_in1k --epochs 30
python train.py --name e4_resnet50_reg      --model resnet50.tv_in1k --epochs 30 --label-smoothing 0.1 --mixup-prob 0.5 --drop-rate 0.3
python train.py --name e5_convnext_reg      --model convnext_tiny.fb_in1k --epochs 30 --label-smoothing 0.1 --mixup-prob 0.5 --drop-rate 0.3
python train.py --name e6_convnext_384      --model convnext_tiny.fb_in1k --img-size 384 --batch-size 16 --epochs 20 --label-smoothing 0.1 --mixup-prob 0.5 --drop-rate 0.3
```

Then:

```
python evaluate.py --run runs/e6_convnext_384     # figures/ and results table
python compare_models.py                          # McNemar's test, E5 vs E6
python export_model.py                            # build deploy/ for the web app
python deploy/app.py                              # run the app locally at http://127.0.0.1:7860
```

## Results (test set, %)

| Experiment | Top-1 | Avg per-class | Top-5 |
|---|---|---|---|
| E1 Scratch CNN | 20.4 | 19.7 | 47.2 |
| E2 ResNet-50 frozen | 68.8 | 68.1 | 90.3 |
| E3 ResNet-50 fine-tuned | 79.5 | 78.9 | 93.6 |
| E4 ResNet-50 + regularisation | 81.4 | 80.9 | 93.0 |
| E5 ConvNeXt-Tiny + regularisation | 86.0 | 85.7 | 96.6 |
| **E6 ConvNeXt-Tiny, 384 px** | **88.7** | **88.4** | **98.0** |
