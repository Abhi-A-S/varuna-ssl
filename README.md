# Varuna-SSL

**Varuna-SSL** explores self-supervised representation learning and ship segmentation in maritime satellite imagery. The project investigates how visual representations learned from unlabelled imagery can support downstream maritime analysis.

The project uses the Airbus Ship Detection Challenge dataset to experiment with SimCLR-based representation learning and a ResNet-18-based segmentation model.

## Project Objectives

- Investigate self-supervised representation learning for maritime satellite imagery.
- Train a SimCLR model to learn visual representations without using ship labels during pretraining.
- Evaluate the learned representations on downstream tasks.
- Develop a ship segmentation model using the learned encoder.
- Analyze segmentation performance across different ship sizes and image-coverage percentages.

## Current Implementation

The project currently includes:

- Dataset downloading, image validation, and annotation analysis utilities.
- Dataset exploration and visualization.
- SimCLR-based self-supervised representation learning.
- Downstream evaluation of learned representations.
- A ResNet-18-based ship segmentation model with a decoder and skip connections.
- Segmentation training using a combination of weighted binary cross-entropy and Dice loss.
- Evaluation using Dice, Intersection over Union (IoU), and ship detection rate.
- Visual inspection of predicted masks and analysis by ground-truth ship coverage.

### Initial Segmentation Results

The full-dataset segmentation baseline was trained for one epoch using 191,756 training images, with 800 images reserved for validation.

At a prediction threshold of 0.50, the validation results were:

| Metric                 |    Result |
| ---------------------- | --------: |
| Pixel-level Dice       |    0.5416 |
| Pixel-level IoU        |    0.3713 |
| Empty-image accuracy   |    98.06% |
| Correctly empty images | 606 / 618 |

These results are an initial baseline, not a final performance claim. Evaluation by ship coverage indicates that segmentation performance varies considerably with ship size, particularly for very small ships.

### Segmentation Performance by Ship Coverage

To investigate how ship size affects segmentation performance, the 800-image validation set was divided into groups based on the percentage of the original image occupied by ground-truth ship masks.

The model was evaluated at a prediction threshold of `0.50`.

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

The `No ships` group is excluded from ship detection rate because it contains no ground-truth ships. Its empty-image performance is reported separately in the overall validation results.

#### Observations

- **Small ships are difficult to segment.** Images with ship coverage below 0.1% show low Dice and IoU scores, indicating missed detections and poor mask overlap.
- **Performance improves with ship coverage.** The 0.5–5% coverage groups achieve Dice scores of approximately 74–76%.
- **Detection does not guarantee accurate segmentation.** The single image in the >5% group has a 100% detection rate but only 13.75% Dice. This group is too small to support general conclusions.
- **Higher-resolution inputs and ship-centred crops** are potential avenues for preserving small-ship details that may be lost during resizing.

These results are an initial analysis of the current segmentation baseline. Further experiments are needed to determine whether alternative input resolutions, training strategies, or self-supervised methods improve performance on small ships.

## Dataset

The project uses the [Airbus Ship Detection Challenge dataset](https://www.kaggle.com/c/airbus-ship-detection/data).

The dataset contains satellite imagery and run-length-encoded (RLE) segmentation annotations for ship masks.

After downloading and extracting the dataset, the expected directory structure is:

```text
raw/
└── airbus-ship-detection/
    ├── train_v2/
    ├── test_v2/
    ├── train_ship_segmentations_v2.csv
    └── sample_submission_v2.csv
```

The test images do not include ground-truth ship masks. Consequently, quantitative segmentation evaluation uses a held-out validation set derived from the labelled training data.

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
git clone <repository-url>
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

| Notebook                                | Purpose                                                               |
| --------------------------------------- | --------------------------------------------------------------------- |
| `01_dataset_exploration.ipynb`          | Dataset inspection, annotation analysis, and visualization.           |
| `02_simclr_baseline.ipynb`              | SimCLR model development and self-supervised pretraining experiments. |
| `03_simclr_baseline_test.ipynb`         | Evaluation and testing of the SimCLR baseline.                        |
| `04_ship_segmentation.ipynb`            | Initial ship segmentation experiments.                                |
| `05_full_training_ship_segment...ipynb` | Full-dataset segmentation training.                                   |
| `06_segmentation_evaluation...ipynb`    | Segmentation evaluation, visualizations, and coverage-based analysis. |

Notebook filenames above are based on the current development layout; replace the shortened names with the exact filenames when documenting the final repository.

## Checkpoints and Artifacts

Model checkpoints are stored in `checkpoints/`.

They include the SimCLR baseline and segmentation experiments trained on different amounts of data. The full-dataset segmentation checkpoint provides the current evaluation baseline.

Checkpoints should be associated with their training configuration, input resolution, threshold, and validation metrics to make comparisons reproducible.

## Limitations and Future Work

Current experiments indicate that small ships are substantially harder to segment than larger ships. Potential improvements include:

- Evaluating higher-resolution inputs and ship-centred crops.
- Training for additional epochs and tracking validation performance.
- Evaluating alternative augmentation strategies for maritime imagery.

These are planned investigations; their effectiveness has not yet been established.

## License and Dataset Terms

This project is an academic course submission. Refer to the dataset provider's terms and conditions for permitted use and redistribution of the Airbus Ship Detection Challenge data.

Third-party dataset files, downloaded imagery, credentials, and local environment files should not be committed to the repository.
