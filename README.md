# RELABELER: A Data-Centric Framework for Detecting and Correcting Corrupted Labels
RELABELER is an end-to-end, data-centric framework for detecting and correcting corrupted labels. 

## Architecture
<img width="1195" height="617" alt="Ảnh màn hình 2026-05-15 lúc 22 24 25" src="https://github.com/user-attachments/assets/3251c34c-2360-4a69-b2a9-dbcd35673cde" />

The figure shows the overview of RELABELER, which consists of two main phases: (i) corrupted label detection and (ii) corrupted label correction.

For corrupted label detection, RELABELER jointly leverages both local and global relationships among data instances to identify potentially noisy samples. 

After detecting suspicious instances, RELABELER further performs label correction by estimating the most probable clean label for each instance based on both its input features and observed noisy label.

#### Deriving the Label Repair Equation via Bayes' Theorem

This section demonstrates how RELABELER derives the label repair equation used in the correction phase from Bayes' theorem.

Specifically, from the fundamental definition of conditional probability, we have:

$$
P(y \mid x, \tilde{y}) = \frac{P(y, x, \tilde{y})}{P(x, \tilde{y})}
$$

Using the chain rule of probability, we can decompose the joint probability in the numerator as:

$$
P(y, x, \tilde{y}) = P(\tilde{y} \mid y, x) \cdot P(y, x)
$$

We can further decompose the term $P(y, x)$ into:

$$
P(y, x) = P(y \mid x) \cdot P(x)
$$

Substituting this back into the numerator's decomposition gives:

$$
P(y, x, \tilde{y}) = P(\tilde{y} \mid y, x) \cdot P(y \mid x) \cdot P(x)
$$

Similarly, the joint probability in the denominator can be expressed as:

$$
P(x, \tilde{y}) = P(\tilde{y} \mid x) \cdot P(x)
$$

By substituting the expanded forms back into the original formula, we get:

$$
P(y \mid x, \tilde{y})
= \frac{P(\tilde{y} \mid y, x) \cdot P(y \mid x) \cdot P(x)}{P(\tilde{y} \mid x) \cdot P(x)}
= \frac{P(\tilde{y} \mid y, x) \cdot P(y \mid x)}{P(\tilde{y} \mid x)}
$$

This is the classic form of Bayes' theorem applied to our context. The term $P(\tilde{y} \mid x)$ in the denominator is the evidence, which acts as a normalization constant. It can be calculated by marginalizing the numerator over all possible clean labels $k \in \{1, \dots, C\}$:

$$
P(\tilde{y} \mid x) = \sum_{k=1}^{C} P(\tilde{y} \mid y=k, x) \cdot P(y=k \mid x)
$$

When our goal is to find the label $y^*$ that maximizes the posterior probability, the denominator $P(\tilde{y} \mid x)$ has the same value for every candidate class $y$. Therefore, for the purpose of the $\arg\max$ operation, this constant term can be disregarded. This simplifies our problem to finding the maximum of the numerator, leading to the following proportionality relationship:

$$
P(y \mid x, \tilde{y}) \propto P(\tilde{y} \mid y, x) \cdot P(y \mid x)
$$

This proportionality relationship is the cornerstone of our correction pipeline. It states that the posterior probability of a clean label is proportional to the product of two key terms: the likelihood of observing the noisy label $\tilde{y}$ given the clean label $y$ and features $x$, and the prior probability of the clean label $y$ given $x$.

## Quick Start
### Prepare Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```
### Dataset 
Datasets can be downloaded from the following link:
https://drive.google.com/drive/folders/1B2GfuWoarV6jTBHU_G_Spdo5wt53cGy9?usp=share_link

### Running
#### Run on Text Data

```bash
python3 scripts/run_relabeler.py \
  --raw-csv-path data/raw/sarcasm_data.csv \
  --text-column text \
  --label-column label \
  --noise-config '{"sym": 0.4}' \
  --output-dir artifacts/outputs/text_run \
  --model-dir artifacts/models
```

#### Run on Image Data

```bash
python3 scripts/run_relabeler.py \
  --raw-image-csv-path data/raw/images.csv \
  --image-path-column image_path \
  --image-label-column label \
  --noise-config '{"sym": 0.4}' \
  --output-dir artifacts/outputs/image_run \
  --model-dir artifacts/models
```

#### Run on Code Data

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

## Contact us 
If you have any questions, comments, or suggestions, please do not hesitate to contact us.
- Email: 22024505@vnu.edu.vn
