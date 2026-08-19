import os
import shutil
import kagglehub
from kagglehub import KaggleDatasetAdapter

DATASET_NAME = "vishalsubbiah/pokemon-images-and-types"
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))


def download_and_sync_dataset(target_dir=DATA_DIR):
    """Download dataset from Kaggle and sync files to local data directory."""
    print(f"Downloading dataset '{DATASET_NAME}' via kagglehub...")
    dataset_path = kagglehub.dataset_download(DATASET_NAME)
    print(f"Dataset downloaded to cache: {dataset_path}")

    os.makedirs(target_dir, exist_ok=True)

    # Sync pokemon.csv
    csv_src = os.path.join(dataset_path, "pokemon.csv")
    csv_dst = os.path.join(target_dir, "pokemon.csv")
    if os.path.exists(csv_src):
        shutil.copy(csv_src, csv_dst)
        print(f"Copied CSV to {csv_dst}")

    # Sync images folder
    images_src = os.path.join(dataset_path, "images")
    images_dst = os.path.join(target_dir, "images")
    if os.path.exists(images_src):
        shutil.copytree(images_src, images_dst, dirs_exist_ok=True)
        img_count = len(os.listdir(images_dst))
        print(f"Copied {img_count} images to {images_dst}")

    return target_dir


def load_pokemon_dataframe(file_path="pokemon.csv"):
    """Load Pokemon dataset as a pandas DataFrame using kagglehub."""
    load_fn = getattr(kagglehub, "dataset_load", getattr(kagglehub, "load_dataset", None))
    df = load_fn(
        KaggleDatasetAdapter.PANDAS,
        DATASET_NAME,
        file_path,
    )
    return df


if __name__ == "__main__":
    print("=== Pokemon Dataset Loader ===")
    df = load_pokemon_dataframe()
    print("\nFirst 5 records:")
    print(df.head())
    print(f"\nTotal records: {len(df)}")

    # Check images
    images_dir = os.path.join(DATA_DIR, "images")
    if os.path.exists(images_dir):
        images = os.listdir(images_dir)
        print(f"Total images available: {len(images)}")
