# Relabeler

Relabeler is an end-to-end pipeline for detecting and correcting label noise:

```text
raw data -> embeddings -> detection phase -> correction phase -> corrected labels
```

The detection phase has a local step and a global step. It only identifies suspected noisy labels; label changes are applied by the correction phase.

## Project Structure

```text
Relabeler/
├── artifacts/
│   ├── models/              # checkpoint and model artifacts
│   └── outputs/             # run outputs, generated embeddings, corrected labels
├── configs/                 # optional config files
├── data/
│   └── raw/                 # raw CSV, image dataset references, or code jsonl data
├── scripts/
│   └── run_relabeler.py     # local wrapper for the package CLI
├── src/
│   ├── cli.py               # command-line entrypoint
│   ├── correction/          # correction phase label updates
│   ├── core/                # pipeline orchestration and utilities
│   ├── data_preprocessing/  # data loading, embedding extraction, synthetic noise
│   ├── detection/           # detection phase local/global steps
│   ├── evaluation/          # metrics
│   └── models/              # neural network modules
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Mac/CPU environments, use `faiss-cpu` as listed in `requirements.txt`. On Kaggle/GPU environments, install a FAISS GPU build that matches the CUDA runtime if needed.


## Run From Raw CSV Text

The CSV file must contain a text column and a label column. The default column names are `text` and `label`.

```bash
python3 scripts/run_relabeler.py \
  --raw-csv-path data/raw/sarcasm_data.csv \
  --text-column text \
  --label-column label \
  --embedding-model-name bert-base-uncased \
  --embedding-batch-size 32 \
  --noise-config '{"sym": 0.4}' \
  --k-neighbors 30 \
  --n-iterations -1 \
  --detection-threshold 0.9 \
  --correction-threshold 0.9 \
  --output-dir artifacts/outputs/sarcasm_run \
  --model-dir artifacts/models
```

## Run From Hugging Face Image Dataset

Image input uses a Hugging Face dataset split. By default, images are read from `image`, labels are read from `label`, and embeddings are generated with CLIP.

```bash
python3 scripts/run_relabeler.py \
  --image-dataset-name beans \
  --image-split train \
  --image-column image \
  --image-label-column labels \
  --embedding-model-name openai/clip-vit-base-patch32 \
  --embedding-batch-size 32 \
  --noise-config '{"sym": 0.4}' \
  --k-neighbors 30 \
  --n-iterations -1 \
  --detection-threshold 0.9 \
  --correction-threshold 0.9 \
  --output-dir artifacts/outputs/image_train_run \
  --model-dir artifacts/models
```

## Run From Raw Code JSONL

The input directory must contain jsonl split files such as `train.jsonl`, `valid.jsonl`, and `test.jsonl`. By default, code is read from `func` and labels are read from `target`; override these with `--code-column` and `--code-label-column` for other schemas.

```bash
python3 scripts/run_relabeler.py \
  --code-jsonl-data-dir data/raw/code_dataset \
  --code-split train \
  --code-column func \
  --code-label-column target \
  --embedding-model-name microsoft/codebert-base \
  --embedding-batch-size 16 \
  --noise-config '{"sym": 0.4}' \
  --k-neighbors 30 \
  --n-iterations -1 \
  --detection-threshold 0.9 \
  --correction-threshold 0.9 \
  --output-dir artifacts/outputs/code_train_run \
  --model-dir artifacts/models
```

## Output

Each run writes results to `--output-dir`:

- `final_corrected_labels.csv`
- `embeddings/` when the input is raw CSV, image, or raw code jsonl data
- `*_label_mapping.json` files that map encoded labels back to the original labels

Temporary checkpoints and model artifacts are written to `--model-dir`.
