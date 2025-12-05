import os
import zipfile
import requests
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
ZIP_PATH = DATA_DIR / "ml-latest-small.zip"

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

def download_dataset():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    if not os.path.exists(ZIP_PATH):
        print("Downloading MovieLens dataset...")
        response = requests.get(MOVIELENS_URL)
        response.raise_for_status()  # raise error on bad status

        with open(ZIP_PATH, "wb") as f:
            f.write(response.content)
        print("Download complete.")
    else:
        print("Dataset already downloaded.")

def extract_dataset():
    extract_path = DATA_DIR / "ml-latest-small"
    if not os.path.exists(extract_path):
        print("Extracting dataset...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print("Extraction complete.")
    else:
        print("Dataset already extracted.")

if __name__ == "__main__":
    download_dataset()
    extract_dataset()