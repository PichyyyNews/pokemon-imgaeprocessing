# pokemon-imgaeprocessing

Image processing and classification project using the Pokemon Images and Types dataset.

## Dataset
- **Source**: [Kaggle - Pokemon Images and Types](https://www.kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types) by Vishal Subbiah
- **Contents**:
  - `pokemon.csv`: Contains metadata including `Name`, `Type1`, `Type2`, and `Evolution`.
  - `images/`: 809 PNG images (120x120 RGBA) across multiple generations.

> [!NOTE]
> Dataset files (`data/`) are ignored in git tracking to keep the repository lightweight. Use `python src/data_loader.py` to download/sync data locally.

## Installation

Install required dependencies:
```bash
pip install -r requirements.txt
```

## Data Preprocessing & Augmentation Pipeline

Complete data cleaning, alpha-to-RGB compositing, stratified splitting, and augmentation pipeline.
- 📘 **Read the Preprocessing Report**: [PREPROCESSING_REPORT.md](PREPROCESSING_REPORT.md)
- 📊 **Dataset Splits**: Saved in [`reports/splits/`](reports/splits/) (Train: 565, Val: 122, Test: 122)

To run the pipeline and generate PyTorch DataLoaders & visualizations:
```bash
python src/preprocessing.py
```

## Exploratory Data Analysis (EDA)

Comprehensive exploratory data analysis has been conducted on image properties and pokemon types.
- 📊 **Read the full report**: [EDA_REPORT.md](EDA_REPORT.md)
- 📈 **Figures and Charts**: Available in [`reports/figures/`](reports/figures/)

To run the EDA analysis and generate visualizations:
```bash
python src/eda.py
```

## Data Loader

Load dataset using `kagglehub` and inspect the data:
```bash
python src/data_loader.py
```
