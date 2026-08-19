# pokemon-imgaeprocessing

Image processing and classification project using the Pokemon Images and Types dataset.

## Dataset
- **Source**: [Kaggle - Pokemon Images and Types](https://www.kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types) by Vishal Subbiah
- **Contents**:
  - `pokemon.csv`: Contains metadata including `Name`, `Type1`, `Type2`, and `Evolution`.
  - `images/`: 809 PNG images of Pokemon across multiple generations.

## Installation

Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Load dataset using `kagglehub` and inspect the data:
```bash
python data_loader.py
```

### Python Example
```python
import kagglehub
from kagglehub import KaggleDatasetAdapter

file_path = "pokemon.csv"

df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "vishalsubbiah/pokemon-images-and-types",
    file_path,
)

print("First 5 records:")
print(df.head())
```
