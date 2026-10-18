# Varuna-SSL

**Varuna-SSL** explores self-supervised representation learning for extracting useful visual features from large volumes of unlabelled maritime imagery.

## Project Objective

Develop and evaluate self-supervised learning methods that learn meaningful representations from maritime satellite imagery without relying on manually assigned labels during pretraining.

The project aims to investigate whether maritime-aware augmentation strategies can improve the quality of learned representations compared with standard self-supervised learning methods.

## Requirements

- Linux development environment.
- Python 3.12.15 or later, as specified in `pyproject.toml`.
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

This creates or updates the project environment, resolves dependencies, and installs the project along with its registered command-line scripts.

Activate the environment if desired:

```bash
source .venv/bin/activate
```

Alternatively, use `uv run` to execute project commands without manually activating the environment.

## Kaggle Authentication

The dataset is hosted on Kaggle and requires API authentication.

1. Sign in to [Kaggle](https://www.kaggle.com/).
2. Open [Kaggle Settings](https://www.kaggle.com/settings).
3. Configure an API token using the authentication method supported by the installed Kaggle CLI.
4. Accept the competition rules.

Verify authentication and dataset access:

```bash
kaggle config view
kaggle competitions files -c airbus-ship-detection
```

The first command displays configuration information. Do not commit API credentials or share authentication tokens.

Official dataset page:

[https://www.kaggle.com/c/airbus-ship-detection/data](https://www.kaggle.com/c/airbus-ship-detection/data)

## Startup Scripts

Run the following commands from the project root:

| Command                   | Description                                             |
| ------------------------- | ------------------------------------------------------- |
| `uv run download-dataset` | Downloads the dataset archive into `./raw`.             |
| `uv run verify-images`    | Verifies image integrity, dimensions, and colour modes. |
| `uv run analyze-dataset`  | Analyzes annotations and checks image-file consistency. |

Extract the downloaded archive before running the verification and analysis commands.

## Project Structure

```text
varuna-ssl/
├── pyproject.toml
├── uv.lock
├── README.md
├── scripts/
│   ├── __init__.py
│   ├── download_dataset.py
│   ├── validate_images.py
│   └── analyze_dataset.py
└── raw/                         # Downloaded and extracted dataset
```

## License and Dataset Terms

This project is an academic course submission. Refer to the dataset provider's terms and conditions for permitted use and redistribution of the Airbus Ship Detection Challenge data.

Third-party dataset files and credentials should not be committed to this repository.
