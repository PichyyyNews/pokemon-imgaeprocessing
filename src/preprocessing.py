import os
import json
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Set seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "pokemon.csv")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
SPLITS_DIR = os.path.join(REPORTS_DIR, "splits")

# All 18 canonical Pokemon types (sorted alphabetically)
ALL_TYPES = sorted([
    "Bug", "Dark", "Dragon", "Electric", "Fairy", "Fighting",
    "Fire", "Flying", "Ghost", "Grass", "Ground", "Ice",
    "Normal", "Poison", "Psychic", "Rock", "Steel", "Water"
])
TYPE_TO_IDX = {t: i for i, t in enumerate(ALL_TYPES)}
IDX_TO_TYPE = {i: t for i, t in enumerate(ALL_TYPES)}

# Standard ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def composite_rgba_to_rgb(img_pil, bg_color="white"):
    """
    Composite RGBA image onto a solid background color and convert to RGB.
    
    bg_color options:
        - "white": (255, 255, 255)
        - "black": (0, 0, 0)
        - "random": random RGB color
        - tuple: specific (R, G, B) tuple
    """
    if img_pil.mode != "RGBA":
        img_pil = img_pil.convert("RGBA")

    if bg_color == "white":
        bg_rgb = (255, 255, 255)
    elif bg_color == "black":
        bg_rgb = (0, 0, 0)
    elif bg_color == "random":
        bg_rgb = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
    elif isinstance(bg_color, (tuple, list)):
        bg_rgb = tuple(bg_color)
    else:
        bg_rgb = (255, 255, 255)

    background = Image.new("RGB", img_pil.size, bg_rgb)
    alpha_channel = img_pil.split()[3]
    background.paste(img_pil, mask=alpha_channel)
    return background


class RandomBackgroundCompositing:
    """Custom transform to composite RGBA image with random solid backgrounds."""
    def __init__(self, p=0.5, bg_range=(180, 255)):
        self.p = p
        self.bg_range = bg_range

    def __call__(self, img_pil):
        if random.random() < self.p:
            bg_color = (
                random.randint(self.bg_range[0], self.bg_range[1]),
                random.randint(self.bg_range[0], self.bg_range[1]),
                random.randint(self.bg_range[0], self.bg_range[1])
            )
            return composite_rgba_to_rgb(img_pil, bg_color=bg_color)
        else:
            return composite_rgba_to_rgb(img_pil, bg_color="white")


def encode_labels(row):
    """
    Encode labels for a single Pokemon row:
    - multi_hot: 18-dim binary vector with 1.0 for Type1 and Type2 (if present)
    - type1_idx: integer index (0..17)
    - type2_idx: integer index (0..17) or -1 if single type
    """
    multi_hot = np.zeros(len(ALL_TYPES), dtype=np.float32)
    t1 = row["Type1"]
    t2 = row["Type2"]

    if t1 in TYPE_TO_IDX:
        multi_hot[TYPE_TO_IDX[t1]] = 1.0
        t1_idx = TYPE_TO_IDX[t1]
    else:
        t1_idx = -1

    if pd.notna(t2) and t2 in TYPE_TO_IDX:
        multi_hot[TYPE_TO_IDX[t2]] = 1.0
        t2_idx = TYPE_TO_IDX[t2]
    else:
        t2_idx = -1

    return multi_hot, t1_idx, t2_idx


def create_stratified_splits(test_size=0.15, val_size=0.15, random_state=SEED):
    """
    Perform stratified split into Train, Validation, and Test sets.
    Saves split dataframes to reports/splits/.
    """
    os.makedirs(SPLITS_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    print(f"Total dataset: {len(df)} records")

    # Composite stratification key based on Type1 to guarantee representation
    strat_key = df["Type1"].copy()
    # For classes with very few samples (e.g., Flying count=3), group with another class for strat split
    counts = strat_key.value_counts()
    rare_classes = counts[counts < 5].index
    strat_key = strat_key.apply(lambda x: "Rare" if x in rare_classes else x)

    # First split: Train+Val (85%) and Test (15%)
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=strat_key
    )

    # Second split: Train (70% total) and Val (15% total -> 15/85 of train_val)
    strat_key_tv = strat_key.loc[train_val_df.index]
    val_rel_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_rel_size,
        random_state=random_state,
        stratify=strat_key_tv
    )

    # Save to CSVs
    train_path = os.path.join(SPLITS_DIR, "train.csv")
    val_path = os.path.join(SPLITS_DIR, "val.csv")
    test_path = os.path.join(SPLITS_DIR, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    summary = {
        "total_samples": len(df),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "train_ratio": round(len(train_df) / len(df), 4),
        "val_ratio": round(len(val_df) / len(df), 4),
        "test_ratio": round(len(test_df) / len(df), 4),
        "all_types": ALL_TYPES,
        "type_to_idx": TYPE_TO_IDX
    }

    with open(os.path.join(SPLITS_DIR, "splits_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Splits created: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df


def get_transforms(image_size=120, is_training=True):
    """Build PyTorch transformation pipelines for train and eval."""
    if is_training:
        return transforms.Compose([
            transforms.Lambda(lambda img: composite_rgba_to_rgb(img, bg_color="random" if random.random() < 0.3 else "white")),
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.04),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Lambda(lambda img: composite_rgba_to_rgb(img, bg_color="white")),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


class PokemonDataset(Dataset):
    """
    PyTorch Dataset for Pokemon Images and Types.
    
    Returns:
        image: torch.Tensor of shape (3, H, W)
        target: multi-hot vector (18,) or single class index (int)
        metadata: dict containing name, type1, type2
    """
    def __init__(self, data_source, images_dir=IMAGES_DIR, transform=None, target_type="multilabel"):
        if isinstance(data_source, str):
            self.df = pd.read_csv(data_source)
        elif isinstance(data_source, pd.DataFrame):
            self.df = data_source.reset_index(drop=True)
        else:
            raise ValueError("data_source must be a file path or pandas DataFrame")

        self.images_dir = images_dir
        self.transform = transform
        self.target_type = target_type

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        name = row["Name"]
        img_path = os.path.join(self.images_dir, f"{name}.png")

        img = Image.open(img_path)

        if self.transform is not None:
            img_tensor = self.transform(img)
        else:
            img_rgb = composite_rgba_to_rgb(img, bg_color="white")
            img_tensor = transforms.ToTensor()(img_rgb)

        multi_hot, t1_idx, t2_idx = encode_labels(row)

        if self.target_type == "multilabel":
            target = torch.tensor(multi_hot, dtype=torch.float32)
        elif self.target_type == "type1":
            target = torch.tensor(t1_idx, dtype=torch.long)
        else:
            target = {
                "multilabel": torch.tensor(multi_hot, dtype=torch.float32),
                "type1": torch.tensor(t1_idx, dtype=torch.long),
                "type2": torch.tensor(t2_idx, dtype=torch.long)
            }

        metadata = {
            "name": name,
            "type1": row["Type1"],
            "type2": row["Type2"] if pd.notna(row["Type2"]) else "None"
        }

        return img_tensor, target, metadata


def get_dataloaders(batch_size=32, image_size=120, target_type="multilabel", num_workers=0):
    """Create Train, Validation, and Test DataLoaders."""
    train_path = os.path.join(SPLITS_DIR, "train.csv")
    val_path = os.path.join(SPLITS_DIR, "val.csv")
    test_path = os.path.join(SPLITS_DIR, "test.csv")

    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
        create_stratified_splits()

    train_dataset = PokemonDataset(train_path, transform=get_transforms(image_size, is_training=True), target_type=target_type)
    val_dataset = PokemonDataset(val_path, transform=get_transforms(image_size, is_training=False), target_type=target_type)
    test_dataset = PokemonDataset(test_path, transform=get_transforms(image_size, is_training=False), target_type=target_type)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def generate_preprocessing_visualizations():
    """Generate visual demonstrations of preprocessing and augmentations."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)

    # 1. Background Compositing Demonstration
    sample_pokemon = ["pikachu", "charizard", "bulbasaur", "gengar"]
    fig, axes = plt.subplots(len(sample_pokemon), 4, figsize=(14, 10))

    bg_variants = [
        ("Original (Transparent RGBA)", lambda img: img),
        ("White Background", lambda img: composite_rgba_to_rgb(img, "white")),
        ("Black Background", lambda img: composite_rgba_to_rgb(img, "black")),
        ("Light Gray Background", lambda img: composite_rgba_to_rgb(img, (230, 230, 230)))
    ]

    for row_idx, name in enumerate(sample_pokemon):
        img_path = os.path.join(IMAGES_DIR, f"{name}.png")
        img = Image.open(img_path)

        for col_idx, (title, func) in enumerate(bg_variants):
            ax = axes[row_idx, col_idx]
            processed = func(img)
            ax.imshow(processed)
            if row_idx == 0:
                ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
            if col_idx == 0:
                ax.set_ylabel(name.capitalize(), fontsize=12, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle("Alpha Channel Compositing onto Various Backgrounds", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    path1 = os.path.join(FIGURES_DIR, "08_preprocessing_compositing.png")
    plt.savefig(path1, dpi=300)
    plt.close()
    print(f"Saved: {path1}")

    # 2. Data Augmentation Variations Showcase
    aug_pipeline = transforms.Compose([
        transforms.Lambda(lambda img: composite_rgba_to_rgb(img, bg_color="random" if random.random() < 0.4 else "white")),
        transforms.RandomResizedCrop(120, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
    ])

    test_subjects = ["squirtle", "lucario", "eevee", "dragonite"]
    fig, axes = plt.subplots(len(test_subjects), 6, figsize=(16, 10))

    for r_idx, name in enumerate(test_subjects):
        img_path = os.path.join(IMAGES_DIR, f"{name}.png")
        img = Image.open(img_path)

        # First column: clean original
        clean_img = composite_rgba_to_rgb(img, "white")
        axes[r_idx, 0].imshow(clean_img)
        axes[r_idx, 0].set_ylabel(name.capitalize(), fontsize=12, fontweight="bold")
        if r_idx == 0:
            axes[r_idx, 0].set_title("Original (Clean)", fontsize=11, fontweight="bold")
        axes[r_idx, 0].set_xticks([])
        axes[r_idx, 0].set_yticks([])

        # Next 5 columns: Augmented variations
        for c_idx in range(1, 6):
            aug_img = aug_pipeline(img)
            ax = axes[r_idx, c_idx]
            ax.imshow(aug_img)
            if r_idx == 0:
                ax.set_title(f"Augmentation #{c_idx}", fontsize=11, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle("Data Augmentation Pipeline Pipeline Samples (Rotation, Flip, Scale, Jitter, Background)",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    path2 = os.path.join(FIGURES_DIR, "09_augmentation_showcase.png")
    plt.savefig(path2, dpi=300)
    plt.close()
    print(f"Saved: {path2}")


def run_pipeline():
    """Main execution of the preprocessing pipeline."""
    print("=== Pokemon Data Preprocessing & Augmentation Pipeline ===")
    
    # 1. Create splits
    train_df, val_df, test_df = create_stratified_splits()

    # 2. Generate visualizations
    print("\nGenerating Preprocessing & Augmentation Visualizations...")
    generate_preprocessing_visualizations()

    # 3. Test PyTorch DataLoaders
    print("\nTesting PyTorch DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=16, image_size=120)
    images, targets, metadata = next(iter(train_loader))

    print(f"Batch image tensor shape: {images.shape}")
    print(f"Batch target tensor shape: {targets.shape}")
    print(f"Batch target min/max: {targets.min().item()}, {targets.max().item()}")
    print("First sample in batch:")
    print(f"  Name: {metadata['name'][0]}")
    print(f"  Type1: {metadata['type1'][0]}, Type2: {metadata['type2'][0]}")
    active_types = [ALL_TYPES[i] for i, val in enumerate(targets[0]) if val > 0.5]
    print(f"  Multi-hot decoded types: {active_types}")

    print("\nPipeline executed successfully!")


if __name__ == "__main__":
    run_pipeline()
