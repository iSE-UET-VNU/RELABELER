# RELABELER: A Data-Centric Framework for Detecting and Correcting Corrupted Labels
RELABELER is an end-to-end, data-centric framework for detecting and correcting corrupted labels. 

# Architecture

The figure shows the overview of RELABELER, which consists of two main phases: (i) corrupted label detection and (ii) corrupted label correction.

For corrupted label detection, RELABELER jointly leverages both local and global relationships among data instances to identify potentially noisy samples. 

After detecting suspicious instances, RELABELER further performs label correction by estimating the most probable clean label for each instance based on both its input features and observed noisy label.

# Quick Start
## Prepare Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running

### Run on Text Data

```bash
python3 scripts/run_relabeler.py \
  --raw-csv-path data/raw/sarcasm_data.csv \
  --text-column text \
  --label-column label \
  --noise-config '{"sym": 0.4}' \
  --output-dir artifacts/outputs/text_run \
  --model-dir artifacts/models
```

### Run on Image Data

```bash
python3 scripts/run_relabeler.py \
  --raw-image-csv-path data/raw/images.csv \
  --image-path-column image_path \
  --image-label-column label \
  --noise-config '{"sym": 0.4}' \
  --output-dir artifacts/outputs/image_run \
  --model-dir artifacts/models
```

### Run on Code Data

```bash
python3 scripts/run_relabeler.py \
  --code-jsonl-data-dir data/raw/code_dataset \
  --code-split train \
  --code-column func \
  --code-label-column target \
  --noise-config '{"sym": 0.4}' \
  --output-dir artifacts/outputs/code_run \
  --model-dir artifacts/models
```

### Outputs

Each run writes the corrected labels and intermediate embeddings to `--output-dir`:

```text
artifacts/outputs/<run_name>/
├── final_corrected_labels.csv
└── embeddings/
    ├── *_embeddings.npy
    ├── *_labels.npy
    └── *_label_mapping.json
```

Temporary model checkpoints are written to `--model-dir`.
