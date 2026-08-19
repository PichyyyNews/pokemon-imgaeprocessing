import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Set paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "pokemon.csv")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

# Official-like colors for Pokemon types
TYPE_COLORS = {
    "Normal": "#A8A878",
    "Fire": "#F08030",
    "Water": "#6890F0",
    "Grass": "#78C850",
    "Electric": "#F8D030",
    "Ice": "#98D8D8",
    "Fighting": "#C03028",
    "Poison": "#A040A0",
    "Ground": "#E0C068",
    "Flying": "#A890F0",
    "Psychic": "#F85888",
    "Bug": "#A8B820",
    "Rock": "#B8A038",
    "Ghost": "#705898",
    "Dragon": "#7038F8",
    "Dark": "#705848",
    "Steel": "#B8B8D0",
    "Fairy": "#EE99AC"
}


def setup_directories():
    """Ensure report and figure directories exist."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


def load_and_augment_dataset():
    """Load metadata and extract comprehensive image features."""
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} records from {CSV_PATH}")

    widths = []
    heights = []
    aspect_ratios = []
    modes = []
    formats = []
    file_sizes_kb = []
    has_alpha = []
    alpha_ratios = []
    mean_r_fg = []
    mean_g_fg = []
    mean_b_fg = []
    brightness_fg = []
    contrast_fg = []
    image_paths = []

    for _, row in df.iterrows():
        img_name = f"{row['Name']}.png"
        img_path = os.path.join(IMAGES_DIR, img_name)
        image_paths.append(img_path)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        size_kb = os.path.getsize(img_path) / 1024.0
        file_sizes_kb.append(size_kb)

        with Image.open(img_path) as img:
            w, h = img.size
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h)
            modes.append(img.mode)
            formats.append(img.format)

            arr = np.array(img)

            # Analyze foreground vs transparency
            if img.mode == "RGBA":
                has_alpha.append(True)
                alpha_channel = arr[:, :, 3]
                fg_mask = alpha_channel > 20
                total_pixels = w * h
                fg_pixels = np.sum(fg_mask)
                alpha_ratio = 1.0 - (fg_pixels / total_pixels)
                alpha_ratios.append(alpha_ratio)

                if fg_pixels > 0:
                    r_vals = arr[:, :, 0][fg_mask]
                    g_vals = arr[:, :, 1][fg_mask]
                    b_vals = arr[:, :, 2][fg_mask]

                    mr = float(np.mean(r_vals))
                    mg = float(np.mean(g_vals))
                    mb = float(np.mean(b_vals))
                    # Perceived brightness (standard photometric formula)
                    bright = 0.299 * mr + 0.587 * mg + 0.114 * mb
                    # Contrast as std of grayscale
                    gray = 0.299 * r_vals + 0.587 * g_vals + 0.114 * b_vals
                    cntrst = float(np.std(gray))

                    mean_r_fg.append(mr)
                    mean_g_fg.append(mg)
                    mean_b_fg.append(mb)
                    brightness_fg.append(bright)
                    contrast_fg.append(cntrst)
                else:
                    mean_r_fg.append(0.0)
                    mean_g_fg.append(0.0)
                    mean_b_fg.append(0.0)
                    brightness_fg.append(0.0)
                    contrast_fg.append(0.0)
            else:
                has_alpha.append(False)
                alpha_ratios.append(0.0)
                if arr.ndim == 3:
                    mr = float(np.mean(arr[:, :, 0]))
                    mg = float(np.mean(arr[:, :, 1]))
                    mb = float(np.mean(arr[:, :, 2]))
                else:
                    mr = mg = mb = float(np.mean(arr))
                bright = 0.299 * mr + 0.587 * mg + 0.114 * mb
                mean_r_fg.append(mr)
                mean_g_fg.append(mg)
                mean_b_fg.append(mb)
                brightness_fg.append(bright)
                contrast_fg.append(float(np.std(arr)))

    df["image_path"] = image_paths
    df["width"] = widths
    df["height"] = heights
    df["aspect_ratio"] = aspect_ratios
    df["mode"] = modes
    df["format"] = formats
    df["file_size_kb"] = file_sizes_kb
    df["has_alpha"] = has_alpha
    df["transparent_ratio"] = alpha_ratios
    df["mean_r"] = mean_r_fg
    df["mean_g"] = mean_g_fg
    df["mean_b"] = mean_b_fg
    df["brightness"] = brightness_fg
    df["contrast"] = contrast_fg
    df["is_dual_type"] = df["Type2"].notna()

    # Save summary csv
    summary_csv = os.path.join(REPORTS_DIR, "eda_summary.csv")
    df.to_csv(summary_csv, index=False)
    print(f"Saved extracted features summary to {summary_csv}")

    return df


def plot_type_distribution(df):
    """Plot distribution of Type 1 and Type 2."""
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    type1_counts = df["Type1"].value_counts()
    colors1 = [TYPE_COLORS.get(t, "#888888") for t in type1_counts.index]
    bars1 = plt.barh(type1_counts.index, type1_counts.values, color=colors1, edgecolor="black", alpha=0.85)
    plt.title("Primary Type (Type 1) Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Number of Pokemon", fontsize=12)
    plt.gca().invert_yaxis()
    for bar in bars1:
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, f"{int(bar.get_width())}",
                 va="center", ha="left", fontsize=10)

    plt.subplot(1, 2, 2)
    type2_counts = df["Type2"].dropna().value_counts()
    colors2 = [TYPE_COLORS.get(t, "#888888") for t in type2_counts.index]
    bars2 = plt.barh(type2_counts.index, type2_counts.values, color=colors2, edgecolor="black", alpha=0.85)
    plt.title("Secondary Type (Type 2) Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Number of Pokemon", fontsize=12)
    plt.gca().invert_yaxis()
    for bar in bars2:
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, f"{int(bar.get_width())}",
                 va="center", ha="left", fontsize=10)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "01_type_distribution.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


def plot_single_vs_dual_type(df):
    """Plot Single vs Dual Type proportion."""
    plt.figure(figsize=(8, 6))
    counts = df["is_dual_type"].value_counts()
    labels = ["Single Type", "Dual Type"]
    colors = ["#4A90E2", "#50E3C2"]
    values = [counts[False], counts[True]]

    plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=140,
            colors=colors, explode=(0.04, 0.04),
            wedgeprops={"edgecolor": "white", "linewidth": 2, "antialiased": True},
            textprops={"fontsize": 12, "weight": "bold"})
    plt.title(f"Pokemon Single vs Dual Type Distribution (Total: {len(df)})", fontsize=14, fontweight="bold")

    path = os.path.join(FIGURES_DIR, "02_single_vs_dual_type.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


def plot_type_co_occurrence(df):
    """Heatmap showing co-occurrence between Type1 and Type2."""
    types = sorted(list(TYPE_COLORS.keys()))
    matrix = pd.crosstab(df["Type1"], df["Type2"]).reindex(index=types, columns=types, fill_value=0)

    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="YlGnBu", cbar=True,
                linewidths=0.5, linecolor="#eeeeee")
    plt.title("Type 1 vs Type 2 Co-occurrence Matrix", fontsize=15, fontweight="bold", pad=12)
    plt.xlabel("Secondary Type (Type 2)", fontsize=12, fontweight="bold")
    plt.ylabel("Primary Type (Type 1)", fontsize=12, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "03_type_co_occurrence_heatmap.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


def plot_image_dimensions_and_filesize(df):
    """Plot distributions of image dimensions and file sizes."""
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    sns.histplot(df["width"], bins=20, color="#3498db", kde=True)
    plt.title("Image Width Distribution", fontsize=12, fontweight="bold")
    plt.xlabel("Width (pixels)")
    plt.ylabel("Frequency")

    plt.subplot(1, 3, 2)
    sns.histplot(df["height"], bins=20, color="#2ecc71", kde=True)
    plt.title("Image Height Distribution", fontsize=12, fontweight="bold")
    plt.xlabel("Height (pixels)")
    plt.ylabel("Frequency")

    plt.subplot(1, 3, 3)
    sns.histplot(df["file_size_kb"], bins=25, color="#e74c3c", kde=True)
    plt.title("File Size (KB) Distribution", fontsize=12, fontweight="bold")
    plt.xlabel("File Size (KB)")
    plt.ylabel("Frequency")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "04_image_dimensions_and_filesize.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


def plot_color_mode_and_transparency(df):
    """Plot color mode counts and transparent pixel ratio distribution."""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    mode_counts = df["mode"].value_counts()
    sns.barplot(x=mode_counts.index, y=mode_counts.values, hue=mode_counts.index, palette="Blues_d", legend=False)
    plt.title("Color Mode Distribution", fontsize=12, fontweight="bold")
    plt.xlabel("Image Mode")
    plt.ylabel("Count")
    for i, v in enumerate(mode_counts.values):
        plt.text(i, v + 5, f"{v} ({v/len(df)*100:.1f}%)", ha="center", fontweight="bold")

    plt.subplot(1, 2, 2)
    sns.histplot(df["transparent_ratio"] * 100, bins=25, color="#9b59b6", kde=True)
    plt.title("Background Transparency Ratio (%)", fontsize=12, fontweight="bold")
    plt.xlabel("Transparent Area (%)")
    plt.ylabel("Pokemon Count")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "05_color_mode_and_transparency.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


def plot_average_color_by_type(df):
    """Plot average foreground RGB color and brightness per Type 1."""
    grouped = df.groupby("Type1").agg({
        "mean_r": "mean",
        "mean_g": "mean",
        "mean_b": "mean",
        "brightness": "mean"
    }).loc[sorted(list(TYPE_COLORS.keys()))]

    plt.figure(figsize=(14, 6))

    # Normalized RGB swatches
    avg_rgb = np.clip(grouped[["mean_r", "mean_g", "mean_b"]].values / 255.0, 0, 1)

    plt.subplot(1, 2, 1)
    bars = plt.barh(grouped.index, grouped["brightness"], color=avg_rgb, edgecolor="black", linewidth=1)
    plt.title("Average Foreground Color & Brightness by Type", fontsize=13, fontweight="bold")
    plt.xlabel("Mean Perceived Brightness (0-255)")
    plt.gca().invert_yaxis()

    plt.subplot(1, 2, 2)
    x = np.arange(len(grouped.index))
    width = 0.25
    plt.bar(x - width, grouped["mean_r"], width=width, label="Red", color="#e74c3c")
    plt.bar(x, grouped["mean_g"], width=width, label="Green", color="#2ecc71")
    plt.bar(x + width, grouped["mean_b"], width=width, label="Blue", color="#3498db")
    plt.xticks(x, grouped.index, rotation=60, ha="right")
    plt.title("RGB Channel Means Across Types", fontsize=13, fontweight="bold")
    plt.ylabel("Mean Intensity (0-255)")
    plt.legend()

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "06_average_color_by_type.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


def plot_sample_gallery(df):
    """Generate a sample visual gallery of Pokemon across various types."""
    sample_types = ["Grass", "Fire", "Water", "Electric", "Psychic", "Dragon", "Ghost", "Bug"]
    samples = []
    for t in sample_types:
        subset = df[df["Type1"] == t]
        if len(subset) > 0:
            samples.append(subset.iloc[0])
            if len(subset) > 1:
                samples.append(subset.iloc[1])

    fig, axes = plt.subplots(2, len(samples) // 2, figsize=(16, 6))
    axes = axes.flatten()

    for idx, (ax, row) in enumerate(zip(axes, samples)):
        img = Image.open(row["image_path"])
        ax.imshow(img)
        title = f"{row['Name'].capitalize()}\n({row['Type1']}"
        if pd.notna(row['Type2']):
            title += f" / {row['Type2']})"
        else:
            title += ")"
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")

    plt.suptitle("Sample Pokemon Images Across Types", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "07_pokemon_sample_gallery.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


def run_eda():
    """Main execution function for EDA."""
    setup_directories()
    df = load_and_augment_dataset()

    print("\n--- Generating Visualizations ---")
    plot_type_distribution(df)
    plot_single_vs_dual_type(df)
    plot_type_co_occurrence(df)
    plot_image_dimensions_and_filesize(df)
    plot_color_mode_and_transparency(df)
    plot_average_color_by_type(df)
    plot_sample_gallery(df)

    print("\nEDA analysis and figure generation completed successfully!")
    return df


if __name__ == "__main__":
    run_eda()
