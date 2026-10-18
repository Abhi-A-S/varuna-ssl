import subprocess
import sys
from pathlib import Path

COMPETITION = "airbus-ship-detection"
RAW_DIR = Path("raw")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    command = [
        "kaggle",
        "competitions",
        "download",
        "-c",
        COMPETITION,
        "-p",
        str(RAW_DIR),
    ]

    print(f"Downloading dataset into {RAW_DIR.resolve()}")

    try:
        subprocess.run(command, check=True)
    except FileNotFoundError:
        print("Error: Kaggle CLI not found. Install it with: uv add kaggle")
        sys.exit(1)
    except subprocess.CalledProcessError as error:
        print(f"Download failed with exit code {error.returncode}")
        sys.exit(error.returncode)

    print("Download command completed successfully.")


if __name__ == "__main__":
    main()
