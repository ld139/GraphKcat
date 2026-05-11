[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](https://github.com/ld139/graphkcat)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-2025.05.18.654694-green)](https://doi.org/10.1101/2025.05.18.654694)

# GraphKcat: Catalytic Pocket-Augmented Learning for Enzyme Kinetic Parameter Prediction

<div align="center">
<img src="./toc.svg" width="600">
</div>

GraphKcat is a structure-aware deep learning framework for predicting enzyme kinetic parameters, including **kcat**, **Km**, and **kcat/Km**. It integrates enzyme sequence representations, substrate molecular representations, environmental variables, and catalytic-pocket structural information.

This repository provides the code for model inference and training, together with processed datasets and a pretrained checkpoint for reproducibility.

---

## Installation

GraphKcat has been tested on Linux systems with GPU support.

```bash
git clone https://github.com/ld139/graphkcat.git
cd graphkcat

conda env create -f apodock.yaml
conda activate apodock
```

Please make sure that `torch`, `torch-scatter`, and `torch-geometric` are compatible with your CUDA version.

---

## Data and Checkpoint

The processed datasets and structural files are available at [Zenodo](https://zenodo.org/records/18501019).

After downloading and extracting the files, the recommended directory structure is:

```text
GraphKcat_data/
├── modeling-datasets/
│   ├── train_dataset_clean_no_structure.csv
│   ├── valid_dataset_clean_no_structure.csv
│   └── test_dataset_clean_no_structure.csv
└── structure_enzyme/
    ├── sample_1/
    ├── sample_2/
    └── ...
```

A pretrained GraphKcat checkpoint is included in this repository:

```text
checkpoint/paper.pt
```

---

## Inference

A pretrained checkpoint is included in this repository for direct prediction:

```text
checkpoint/paper.pt
```

A demo input file is provided at:

```text
example/test.csv
```

### Input CSV format

For inference, prepare a CSV file containing the following columns:

| Column | Description |
|---|---|
| `id` | Unique identifier for each enzyme-substrate pair |
| `Smiles` | Substrate SMILES string |
| `protein` | Path to the protein structure file |
| `ligand` | Path to the ligand structure file |
| `organism` | Source organism |
| `pH` | Reaction pH |
| `temperature` | Reaction temperature |

Example input:

| id | Smiles | protein | ligand | organism | pH | temperature |
|---|---|---|---|---|---:|---:|
| example_001 | CCO | example/structure_example/example_001_protein.pdb | example/structure_example/example_001_ligand.sdf | Escherichia coli | 7.0 | 298.15 |

If an enzyme-substrate complex structure is already available, the CSV file can additionally include:

| Optional column | Description |
|---|---|
| `complex` | Path to the enzyme-substrate complex structure file |

---

### Run prediction

```bash
python predict.py \
    --csv_file example/test.csv \
    --output_dir output/inference \
    --cpkt_path checkpoint/paper.pt
```

The prediction results will be saved to:

```text
output/inference/predictions_*.csv
```

### Output CSV format

| Column | Description |
|---|---|
| `id` | Unique identifier from the input file |
| `pred_log_kcat_graphkcat` | Predicted log(kcat) |
| `pred_log_km_graphkcat` | Predicted log(Km) |
| `pred_log_kcat_km_graphkcat` | Predicted log(kcat/Km), calculated as log(kcat) - log(Km) |

Example output:

| id | pred_log_kcat_graphkcat | pred_log_km_graphkcat | pred_log_kcat_km_graphkcat |
|---|---:|---:|---:|
| example_001 | 1.23 | -0.45 | 1.68 |

---

## Training

The default training configuration is defined in:

```text
config/config_dict.py
```

The training script uses the configuration `TrainConfig_kcat_enz` and reads the processed datasets from:

```text
<data_root>/modeling-datasets/
```

Specifically, it expects:

```text
train_dataset_clean_no_structure.csv
valid_dataset_clean_no_structure.csv
test_dataset_clean_no_structure.csv
```

To train GraphKcat, first set `data_root` in:

```text
config/config_dict.py
```

For example:

```python
"data_root": "/path/to/GraphKcat_data"
```

Then run:

```bash
python train.py
```

---

## Citation

If you use GraphKcat, please cite:

```text
GraphKcat: Catalytic pocket-informed augmentation of enzyme kinetic parameters prediction via hierarchical graph learning.
bioRxiv 2025.05.18.654694.
https://doi.org/10.1101/2025.05.18.654694
```

---

## License

This project is released under the MIT License.