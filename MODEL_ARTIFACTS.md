# Varuna-SSL — Model Artifacts

This document describes the trained models and experiment artifacts released as part of the Varuna-SSL project.

The model checkpoints are distributed separately from the source repository through GitHub Releases.

## Model Release

**Release:** [Varuna-SSL — Model Checkpoints v1](https://github.com/Abhi-A-S/varuna-ssl/releases/tag/models-v1)

**Download:** `varuna-ssl-models-v1.zip`

The archive contains the following structure:

```text
checkpoints/
├── simclr_best.pt
├── simclr_classifier_best.pt
├── varuna_segmentation_full_dataset_epoch1.pt
├── simclr_training_history.csv
├── simclr_step_history.csv
├── simclr_test_results.csv
└── simclr_classifier_history.csv
```

## 1. SimCLR Pretrained Model

**Checkpoint:** `simclr_best.pt`

### Architecture

- Encoder: ResNet-18
- Projection head: 512 → 512 → 128
- Projection dimension: 128
- Loss function: NT-Xent
- Temperature: 0.5
- Input resolution: 224 × 224

### Training

- Dataset: Airbus Ship Detection Challenge
- Training images: 134,789
- Epochs: 5
- Batch size: 32
- Optimizer: AdamW
- Learning rate: 0.0003

The model was trained using self-supervised contrastive learning with augmented image pairs. Ship annotations were not used during pretraining.

### Purpose

The checkpoint contains the ResNet-18 encoder and projection head. The encoder can be used to extract image representations for downstream tasks.

## 2. Fine-Tuned Ship Classifier

**Checkpoint:** `simclr_classifier_best.pt`

### Architecture

- Encoder: ResNet-18 initialized from the SimCLR checkpoint.
- Classification head: 512 → 256 → 2.
- Activation: ReLU.
- Dropout: 0.3.
- Output classes: Ship and No Ship.
- Input resolution: 224 × 224.

### Training

- Initialization: `simclr_best.pt`
- Epochs: 10
- Batch size: 32
- Optimizer: AdamW
- Encoder learning rate: 0.00001
- Classifier learning rate: 0.001
- Loss function: Weighted Cross Entropy

### Evaluation Results

| Metric                 | Result |
| ---------------------- | -----: |
| Test accuracy          | 93.77% |
| Test Average Precision | 94.09% |
| Test ROC-AUC           | 0.9808 |
| Ship precision         | 82.35% |
| Ship recall            | 91.40% |
| Ship F1-score          | 86.64% |

### Purpose

The fine-tuned model performs binary image-level classification to determine whether a satellite image contains a ship.

It does not produce ship locations or segmentation masks.

## 3. Ship Segmentation Model

**Checkpoint:** `varuna_segmentation_full_dataset_epoch1.pt`

### Architecture

- Encoder: ResNet-18.
- Decoder: Convolutional upsampling blocks.
- Skip connections: Encoder-to-decoder feature connections.
- Output: Single-channel segmentation mask.
- Input resolution: 256 × 256.
- Upsampling: Bilinear interpolation.

The segmentation encoder was initialized with randomly initialized ResNet-18 weights. It was not initialized from the trained SimCLR checkpoint.

### Training

- Dataset: Airbus Ship Detection Challenge.
- Training images: 191,756.
- Epochs: 1.
- Batch size: 16.
- Optimizer: AdamW.
- Learning rate: 0.0001.
- Loss: Weighted Binary Cross Entropy + Dice Loss.

### Evaluation Results

The training notebook reported the following results on its stratified 800-image validation split:

| Metric           | Result |
| ---------------- | -----: |
| Pixel-level Dice | 0.6242 |
| Pixel-level IoU  | 0.4537 |

A separate evaluation notebook loaded the saved checkpoint and evaluated it on a separately constructed 800-image validation split:

| Metric               | Result |
| -------------------- | -----: |
| Pixel-level Dice     | 0.5416 |
| Pixel-level IoU      | 0.3713 |
| Empty-image accuracy | 98.06% |

These results come from different validation image sets and should not be treated as a direct comparison.

### Purpose

The segmentation model produces pixel-level predictions identifying ship regions in maritime satellite imagery.

## 4. Experiment History Files

| File                            | Description                                            |
| ------------------------------- | ------------------------------------------------------ |
| `simclr_training_history.csv`   | Epoch-level SimCLR training and validation metrics.    |
| `simclr_step_history.csv`       | Step-level SimCLR training history.                    |
| `simclr_test_results.csv`       | SimCLR linear-probe test evaluation results.           |
| `simclr_classifier_history.csv` | Fine-tuned classifier training and validation history. |

## 5. Loading the Models

The checkpoints are PyTorch state dictionaries stored in checkpoint dictionaries. They are intended to be loaded using the corresponding model architecture definitions from the project notebooks.

The architecture definitions are available in:

| Model                 | Notebook                                                                          |
| --------------------- | --------------------------------------------------------------------------------- |
| SimCLR                | `07_simclr_full_dataset.ipynb`                                                    |
| Fine-tuned classifier | `09_finetune_simclr_classifier.ipynb`                                             |
| Ship segmentation     | `05_full_training_ship_segmentation.ipynb` and `06_segmentation_evaluation.ipynb` |

For example, the fine-tuned classifier checkpoint can be loaded using its corresponding model class:

```python
import torch

checkpoint = torch.load(
    "checkpoints/simclr_classifier_best.pt",
    map_location="cpu",
    weights_only=False,
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

Here, `model` must first be instantiated using the `SimCLRClassifier` architecture defined in the fine-tuning notebook.

The same principle applies to the other checkpoints: instantiate the matching architecture before loading its `model_state_dict`.

## 6. Reproducibility

To reproduce the experiments:

1. Clone the repository.
2. Install dependencies using `uv sync`.
3. Download and prepare the Airbus Ship Detection Challenge dataset.
4. Download the model release if pretrained checkpoints are required.
5. Extract the release archive into the project root.
6. Follow the corresponding notebook workflow.

Refer to the main [README](README.md) for environment setup, dataset preparation, training configurations, and experiment results.
