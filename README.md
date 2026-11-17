# Varuna-SSL

**Varuna-SSL** explores self-supervised representation learning and ship segmentation in maritime satellite imagery. The project investigates how visual representations learned from unlabelled imagery can support downstream maritime analysis.

The project uses the Airbus Ship Detection Challenge dataset and explores SimCLR-based representation learning with a ResNet-18 encoder, alongside a ResNet-18-based ship segmentation model.

## Project Objectives

- Investigate self-supervised representation learning for maritime satellite imagery.
- Train a SimCLR model to learn visual representations without using ship labels during pretraining.
- Evaluate learned representations on downstream ship classification tasks.
- Develop a ship segmentation model using a ResNet-18-based architecture.
- Analyze model performance across different ship sizes and image-coverage percentages.

## Current Implementation

The project currently includes:

- Dataset downloading, image validation, and annotation analysis utilities.
- Dataset exploration and visualization.
- Dataset splitting with stratified train, validation, and test sets.
- SimCLR-based self-supervised representation learning.
- Downstream evaluation using a linear classifier trained on frozen representations.
- Analysis of classification errors and false negatives.
- Ship coverage-based evaluation to understand the effect of ship size on detection.
- A ResNet-18-based ship segmentation model with a decoder and skip connections.
- Segmentation training using weighted binary cross-entropy and Dice loss.
- Segmentation evaluation using Dice, Intersection over Union (IoU), and ship detection rate.

## SimCLR Self-Supervised Learning

SimCLR is used to learn visual representations from maritime satellite images without using ship annotations during pretraining.

The model uses a ResNet-18 encoder and a projection head. Two augmented views of each image are passed through the network, and the NT-Xent contrastive loss encourages representations of the same image to be similar while separating representations of different images.

### Model Architecture

- **Encoder:** ResNet-18, initialized without pretrained ImageNet weights.
- **Projection head:** Fully connected layers mapping the encoder representation to a 128-dimensional projection.
- **Loss function:** Normalized Temperature-Scaled Cross Entropy (NT-Xent).
- **Temperature:** 0.5.
- **Input resolution:** 224 × 224 pixels.

### Training Configuration

| Parameter            |           Value |
| -------------------- | --------------: |
| Training images      |         134,789 |
| Batch size           |              32 |
| Epochs               |               5 |
| Optimizer            |           AdamW |
| Learning rate        |          0.0003 |
| Weight decay         |          0.0001 |
| Projection dimension |             128 |
| Temperature          |             0.5 |
| Mixed precision      | Enabled on CUDA |

The encoder was trained on the full training split using augmented image pairs. Ship labels were not used during contrastive pretraining.

### SimCLR Training Results

The model was trained for five epochs. Training loss decreased throughout training, while validation performance was monitored using a linear-probe evaluation.

_Refer to `checkpoints/simclr_training_history.csv` for the complete training history._

The best validation PR-AUC of **0.8387** was achieved at epoch 5.

### Downstream Linear-Probe Evaluation

To evaluate the learned representations, the SimCLR encoder was frozen and image features were extracted. A separate linear classifier was trained using these features to distinguish between ship and no-ship images.

The held-out test set contained 28,884 images, including 6,384 ship-containing images and 22,500 images without ships.

| Metric    | No Ship |   Ship |
| --------- | ------: | -----: |
| Precision |  0.9459 | 0.6831 |
| Recall    |  0.8920 | 0.8203 |
| F1-score  |  0.9182 | 0.7455 |

| Overall Metric    | Result |
| ----------------- | -----: |
| Accuracy          | 87.62% |
| Macro F1-score    | 0.8318 |
| Weighted F1-score | 0.8800 |

The confusion matrix was:

|                | Predicted No Ship | Predicted Ship |
| -------------- | ----------------: | -------------: |
| Actual No Ship |            20,071 |          2,429 |
| Actual Ship    |             1,147 |          5,237 |

These results show that the learned representations contain useful information for downstream ship classification. However, the classifier performs better on no-ship images than on ship-containing images.

### Analysis of Ship Coverage

The test set was further analyzed by grouping ship-containing images according to the percentage of the original image occupied by ground-truth ship masks.

| Ship Coverage | Images | Correctly Detected | Missed | Recall (%) |
| ------------- | -----: | -----------------: | -----: | ---------: |
| <0.01%        |    375 |                277 |     98 |      73.87 |
| 0.01–0.05%    |  1,536 |              1,086 |    450 |      70.70 |
| 0.05–0.1%     |    613 |                476 |    137 |      77.65 |
| 0.1–0.5%      |  1,894 |              1,597 |    297 |      84.32 |
| 0.5–1%        |    884 |                803 |     81 |      90.84 |
| 1–2%          |    748 |                688 |     60 |      91.98 |
| 2–5%          |    327 |                303 |     24 |      92.66 |
| >5%           |      7 |                  7 |      0 |     100.00 |

#### Observations

- Ship-containing images with lower coverage are more difficult to classify correctly.
- Recall generally improves as ship coverage increases.
- Images with coverage below 0.1% have lower detection recall than images containing larger ships.
- The results suggest that small ships and limited visual features remain challenging for the learned representations.
- The group containing more than 5% ship coverage has only seven images, so its result should not be generalized.

### Representation Analysis

Additional analysis was performed on the learned feature representations, including:

- Feature extraction from the frozen SimCLR encoder.
- Linear classification using the extracted features.
- Principal Component Analysis (PCA) for dimensionality reduction and visualization.
- Class prototype similarity analysis.
- False-negative inspection.
- Analysis of classification performance across ship-coverage groups.

These analyses help investigate how the learned representations distinguish ship-containing images from images without ships.

## Ship Segmentation

The project also includes a ResNet-18-based ship segmentation model with a decoder and skip connections.

The segmentation model was trained using a combination of weighted binary cross-entropy and Dice loss. Its predictions were evaluated against ground-truth ship masks.

### Initial Segmentation Results

The full-dataset segmentation baseline was trained for one epoch using 191,756 training images, with 800 images reserved for validation.

At a prediction threshold of `0.50`, the validation results were:

| Metric                 |    Result |
| ---------------------- | --------: |
| Pixel-level Dice       |    0.5416 |
| Pixel-level IoU        |    0.3713 |
| Empty-image accuracy   |    98.06% |
| Correctly empty images | 606 / 618 |

These results are an initial baseline, not a final performance claim. Evaluation by ship coverage indicates that segmentation performance varies considerably with ship size, particularly for very small ships.

### Segmentation Performance by Ship Coverage

| Ship Coverage | Images | Dice (%) | IoU (%) | Ship Detection (%) |
| ------------- | -----: | -------: | ------: | -----------------: |
| No ships      |    618 |      N/A |     N/A |                N/A |
| 0–0.01%       |      8 |     0.00 |    0.00 |              25.00 |
| 0.01–0.05%    |     44 |     3.64 |    1.85 |              38.64 |
| 0.05–0.1%     |     16 |     9.32 |    4.89 |              56.25 |
| 0.1–0.5%      |     70 |    55.05 |   37.98 |              84.29 |
| 0.5–1%        |     20 |    74.90 |   59.87 |             100.00 |
| 1–2%          |     13 |    73.66 |   58.30 |              92.31 |
| 2–5%          |     10 |    75.59 |   60.76 |              90.00 |
| >5%           |      1 |    13.75 |    7.38 |             100.00 |

**Metrics:**

- **Dice:** Measures pixel-level overlap between predicted and ground-truth masks.
- **IoU:** Measures the intersection of predicted and ground-truth masks relative to their union.
- **Ship Detection:** Percentage of ship-containing images in a coverage group where the model predicted at least one ship pixel.

The `No ships` group is excluded from ship detection rate because it contains no ground-truth ships. Its empty-image performance is reported separately.

#### Observations

- Small ships are difficult to segment. Images with ship coverage below 0.1% show low Dice and IoU scores.
- Performance generally improves with ship coverage.
- Detection does not guarantee accurate segmentation. The single image in the >5% group has a 100% detection rate but only 13.75% Dice.
- Higher-resolution inputs and ship-centred crops are potential approaches for preserving small-ship details.

These results are an initial analysis of the segmentation baseline. Further experiments are needed to establish whether alternative input resolutions and training strategies improve performance.

## Dataset

The project uses the [Airbus Ship Detection Challenge dataset](https://www.kaggle.com/c/airbus-ship-detection/data).

The dataset contains satellite imagery and run-length-encoded (RLE) segmentation annotations for ship masks.

### Dataset Statistics

| Description                     |   Count |
| ------------------------------- | ------: |
| Unique labelled training images | 192,556 |
| Images containing ships         |  42,556 |
| Images without ships            | 150,000 |
| Annotation rows                 | 231,723 |
| Non-empty mask rows             |  81,723 |
| Empty mask rows                 | 150,000 |

### Dataset Split

The labelled images were divided into stratified training, validation, and test sets.

| Split      | Total Images |     No Ship |       Ship |
| ---------- | -----------: | ----------: | ---------: |
| Training   |      134,789 |     105,000 |     29,789 |
| Validation |       28,883 |      22,500 |      6,383 |
| Test       |       28,884 |      22,500 |      6,384 |
| **Total**  |  **192,556** | **150,000** | **42,556** |

The test images used for internal evaluation are held-out labelled images from the original training dataset. The competition's separate `test_v2` images do not include ground-truth ship masks.

After downloading and extracting the dataset, the expected directory structure is:

```text
raw/
└── airbus-ship-detection/
    ├── train_v2/
    ├── test_v2/
    ├── train_ship_segmentations_v2.csv
    └── sample_submission_v2.csv
```

## Requirements

- Linux development environment.
- Python 3.12.15, as specified in `pyproject.toml`.
- [uv](https://docs.astral.sh/uv/).
- NVIDIA GPU with a compatible driver for CUDA-accelerated training.
- Kaggle account with access to the Airbus Ship Detection Challenge.
- Sufficient storage for the dataset and model checkpoints.

## Environment Setup

Clone the repository and enter the project directory:

```bash
git clone https://github.com/Abhi-A-S/varuna-ssl.git
cd varuna-ssl
```

Install the project and its dependencies:

```bash
uv sync
```

Alternatively, activate the virtual environment:

```bash
source .venv/bin/activate
```

You can also execute registered project commands directly through `uv run` without manually activating the environment.

## Kaggle Authentication

1. Sign in to [Kaggle](https://www.kaggle.com/).
2. Open [Kaggle Settings](https://www.kaggle.com/settings).
3. Configure API authentication using the method supported by your installed Kaggle CLI.
4. Accept the competition rules.

Verify authentication and dataset access:

```bash
kaggle config view
kaggle competitions files -c airbus-ship-detection
```

Do not commit API credentials or share authentication tokens.

## Dataset Utilities

Run these commands from the project root:

| Command                   | Description                                             |
| ------------------------- | ------------------------------------------------------- |
| `uv run download-dataset` | Downloads the dataset archive into `./raw`.             |
| `uv run verify-images`    | Verifies image integrity, dimensions, and colour modes. |
| `uv run analyze-dataset`  | Analyzes annotations and checks image-file consistency. |

Extract the downloaded archive before running the image verification and dataset analysis commands.

## Notebook Workflow

The notebooks contain the exploratory analysis, model development, training, and evaluation experiments.

| Notebook                                | Purpose                                                                    |
| --------------------------------------- | -------------------------------------------------------------------------- |
| `01_dataset_exploration.ipynb`          | Dataset inspection, annotation analysis, and visualization.                |
| `02_simclr_baseline.ipynb`              | Earlier SimCLR model development experiments.                              |
| `03_simclr_baseline_test.ipynb`         | Earlier SimCLR evaluation and testing.                                     |
| `04_ship_segmentation.ipynb`            | Initial ship segmentation experiments.                                     |
| `05_full_training_ship_segment...ipynb` | Full-dataset segmentation training.                                        |
| `06_segmentation_evaluation...ipynb`    | Segmentation evaluation and coverage-based analysis.                       |
| `07_simclr_full_dataset.ipynb`          | Full-dataset SimCLR pretraining using the final retained baseline.         |
| `08_simclr_model_analysis.ipynb`        | Downstream evaluation, feature analysis, and false-negative investigation. |

The notebook filenames for the segmentation experiments should be checked against the repository before finalizing this table.

## Checkpoints and Artifacts

Model checkpoints are stored in `checkpoints/`.

The retained SimCLR checkpoints and training history are:

```text
checkpoints/
├── simclr_best.pt
├── simclr_latest.pt
└── simclr_training_history.csv
```

- `simclr_best.pt` contains the best-performing SimCLR checkpoint.
- `simclr_latest.pt` contains the latest training checkpoint.
- `simclr_training_history.csv` contains the recorded training metrics.

The SimCLR checkpoint contains the encoder and projection head. The downstream linear classifier is trained separately during evaluation and is not part of the SimCLR checkpoint.

Segmentation checkpoints and other artifacts may also be stored in this directory.

## Limitations and Future Work

Current experiments indicate that small ships are challenging for both representation-based classification and segmentation.

Potential future investigations include:

- Evaluating higher-resolution inputs and ship-centred crops.
- Investigating alternative augmentation strategies for maritime imagery.
- Exploring improved downstream classification and segmentation approaches.
- Evaluating whether the learned representations can support more detailed maritime analysis.

The filtered SimCLR training experiment, which excluded ship-containing images below 0.1% mask coverage, was discontinued because it did not improve downstream performance over the full-dataset baseline. The original full-dataset SimCLR model is retained as the project's baseline.

## License and Dataset Terms

This project is an academic research project. Refer to the dataset provider's terms and conditions for permitted use and redistribution of the Airbus Ship Detection Challenge data.

Third-party dataset files, downloaded imagery, credentials, and local environment files should not be committed to the repository.
