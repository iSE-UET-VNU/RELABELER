from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

from core.utils import ensure_dir, get_device
from data_preprocessing.label_encoding import encode_labels, save_label_mapping

DEFAULT_IMAGE_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_TEXT_MODEL = "bert-base-uncased"
DEFAULT_CODE_MODEL = "microsoft/codebert-base"


def get_save_path(dataset_name: str, encode_model: str, output_dir: str | Path = ".") -> Path:
    encode_model_filename = encode_model.replace("/", "-")
    dataset_dirname = dataset_name.replace("/", "-")
    path = Path(output_dir) / f"{dataset_dirname}-{encode_model_filename}"
    return ensure_dir(path)


def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def _resolve_column(sample: dict, column: str, fallbacks: tuple[str, ...] = ()) -> str:
    if column in sample:
        return column

    for fallback in fallbacks:
        if fallback in sample:
            return fallback

    available = ", ".join(sample.keys())
    raise KeyError(f"Column '{column}' not found. Available columns: {available}")


def _embedding_from_outputs(outputs, attention_mask=None, output_attr: str | None = None):
    if output_attr and hasattr(outputs, output_attr):
        value = getattr(outputs, output_attr)
        if value is not None:
            return value

    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output

    if attention_mask is not None and hasattr(outputs, "last_hidden_state"):
        return mean_pooling(outputs.last_hidden_state, attention_mask)

    return outputs.last_hidden_state.mean(dim=1)


def extract_img_embedding(
    dataset_name: str,
    batch_size: int = 32,
    encode_model: str = DEFAULT_IMAGE_MODEL,
    output_dir: str | Path = ".",
    splits: tuple[str, ...] = ("train", "test"),
    image_column: str = "image",
    label_column: str = "label",
    device: str | torch.device | None = None,
):
    """Extract image embeddings from any Hugging Face image dataset.

    Defaults to CLIP for image encoding. Dataset-specific schemas are handled
    through ``image_column`` and ``label_column`` instead of hard-coded dataset
    names.
    """
    device = get_device(str(device) if device else None)
    dataset = load_dataset(dataset_name)
    processor = AutoImageProcessor.from_pretrained(encode_model)
    model = AutoModel.from_pretrained(encode_model).to(device)
    model.eval()

    def embed_img_batch(batch_images):
        inputs = processor(images=batch_images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = _embedding_from_outputs(outputs, output_attr="image_embeds")
        return embeddings.cpu().numpy()

    split_specs = {}
    all_labels = []
    for split_name in splits:
        split_data = dataset[split_name]
        if len(split_data) == 0:
            raise ValueError(f"Split '{split_name}' is empty.")

        sample = split_data[0]
        resolved_image_column = _resolve_column(sample, image_column, fallbacks=("img",))
        resolved_label_column = _resolve_column(sample, label_column)
        labels = [ex[resolved_label_column] for ex in split_data]
        split_specs[split_name] = (split_data, resolved_image_column, labels)
        all_labels.extend(labels)

    _, class_labels = encode_labels(all_labels)

    def process_split(split_name):
        split_data, resolved_image_column, labels = split_specs[split_name]

        images = [ex[resolved_image_column] for ex in split_data]
        encoded_labels, _ = encode_labels(labels, class_labels=class_labels)

        all_embeddings = []
        for i in tqdm(range(0, len(images), batch_size), desc=f"Processing {split_name}"):
            batch_images = images[i : i + batch_size]
            all_embeddings.append(embed_img_batch(batch_images))

        return np.vstack(all_embeddings), encoded_labels

    save_path = get_save_path(dataset_name, encode_model, output_dir)
    save_label_mapping(save_path / "label_mapping.json", class_labels)
    outputs = {}
    for split in splits:
        embeddings, labels = process_split(split)
        np.save(save_path / f"features_{split}.npy", embeddings)
        np.save(save_path / f"labels_{split}.npy", labels)
        outputs[split] = (embeddings, labels)

    return outputs


def embed_hf_image_split(
    dataset_name: str,
    output_dir: str | Path,
    split: str = "train",
    model_name: str = DEFAULT_IMAGE_MODEL,
    image_column: str = "image",
    label_column: str = "label",
    batch_size: int = 32,
    output_prefix: str = "clip",
    device: str | torch.device | None = None,
):
    device = get_device(str(device) if device else None)
    output_dir = ensure_dir(output_dir)
    dataset = load_dataset(dataset_name)
    split_data = dataset[split]
    if len(split_data) == 0:
        raise ValueError(f"Split '{split}' is empty.")

    sample = split_data[0]
    resolved_image_column = _resolve_column(sample, image_column, fallbacks=("img",))
    resolved_label_column = _resolve_column(sample, label_column)

    images = [ex[resolved_image_column] for ex in split_data]
    labels, class_labels = encode_labels([ex[resolved_label_column] for ex in split_data])

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    def embed_img_batch(batch_images):
        inputs = processor(images=batch_images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = _embedding_from_outputs(outputs, output_attr="image_embeds")
        return embeddings.cpu().numpy()

    embeddings = []
    print(f"Processing {len(images)} images for split {split}...")
    for i in tqdm(range(0, len(images), batch_size), desc=split):
        embeddings.append(embed_img_batch(images[i : i + batch_size]))

    embeddings = np.vstack(embeddings)
    embeddings_path = output_dir / f"{output_prefix}_{split}_embeddings.npy"
    labels_path = output_dir / f"{output_prefix}_{split}_labels.npy"
    np.save(embeddings_path, embeddings)
    np.save(labels_path, labels)
    mapping_path = save_label_mapping(output_dir / f"{output_prefix}_{split}_label_mapping.json", class_labels)

    print(f"{split} image embeddings saved: {embeddings_path} shape={embeddings.shape}")
    print(f"{split} labels saved: {labels_path} shape={labels.shape}")
    print(f"{split} label mapping saved: {mapping_path}")
    return embeddings_path, labels_path


def extract_text_embedding(
    dataset_name: str,
    batch_size: int = 32,
    encode_model: str = DEFAULT_TEXT_MODEL,
    output_dir: str | Path = ".",
    splits: tuple[str, ...] = ("train", "test"),
    text_column: str = "text",
    label_column: str = "label",
    max_length: int = 512,
    device: str | torch.device | None = None,
):
    """Extract text embeddings from any Hugging Face text dataset.

    Defaults to BERT for text encoding. Use ``text_column`` and
    ``label_column`` to adapt to a dataset's schema.
    """
    device = get_device(str(device) if device else None)
    dataset = load_dataset(dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(encode_model)
    model = AutoModel.from_pretrained(encode_model).to(device)
    model.eval()

    def embed_text_batch(batch_texts):
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = _embedding_from_outputs(outputs, attention_mask=inputs.get("attention_mask"))
        return embeddings.cpu().numpy()

    split_specs = {}
    all_labels = []
    for split_name in splits:
        split_data = dataset[split_name]
        if len(split_data) == 0:
            raise ValueError(f"Split '{split_name}' is empty.")

        sample = split_data[0]
        resolved_text_column = _resolve_column(sample, text_column)
        resolved_label_column = _resolve_column(sample, label_column)
        labels = [ex[resolved_label_column] for ex in split_data]
        split_specs[split_name] = (split_data, resolved_text_column, labels)
        all_labels.extend(labels)

    _, class_labels = encode_labels(all_labels)

    def process_split(split_name):
        split_data, resolved_text_column, labels = split_specs[split_name]

        texts = [str(ex[resolved_text_column]) for ex in split_data]
        encoded_labels, _ = encode_labels(labels, class_labels=class_labels)

        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Processing {split_name}"):
            batch_texts = texts[i : i + batch_size]
            all_embeddings.append(embed_text_batch(batch_texts))

        return np.vstack(all_embeddings), encoded_labels

    save_path = get_save_path(dataset_name, encode_model, output_dir)
    save_label_mapping(save_path / "label_mapping.json", class_labels)
    outputs = {}
    for split in splits:
        embeddings, labels = process_split(split)
        np.save(save_path / f"features_{split}.npy", embeddings)
        np.save(save_path / f"labels_{split}.npy", labels)
        outputs[split] = (embeddings, labels)

    return outputs


def embed_csv_text(
    csv_file_path: str | Path,
    output_dir: str | Path,
    text_column: str = "text",
    label_column: str = "label",
    model_name: str = DEFAULT_TEXT_MODEL,
    batch_size: int = 32,
    output_prefix: str = "bert",
    max_length: int = 512,
    device: str | torch.device | None = None,
):
    device = get_device(str(device) if device else None)
    output_dir = ensure_dir(output_dir)

    df = pd.read_csv(csv_file_path)
    df = df.dropna(subset=[text_column, label_column])
    texts = df[text_column].astype(str).tolist()
    labels, class_labels = encode_labels(df[label_column].to_numpy())

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    def embed_text_batch(batch_texts):
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = _embedding_from_outputs(outputs, attention_mask=inputs.get("attention_mask"))
        return embeddings.cpu().numpy()

    embeddings = []
    print(f"Processing {len(texts)} texts...")
    for i in tqdm(range(0, len(texts), batch_size)):
        embeddings.append(embed_text_batch(texts[i : i + batch_size]))

    embeddings = np.vstack(embeddings)
    embeddings_path = output_dir / f"{output_prefix}_embeddings.npy"
    labels_path = output_dir / f"{output_prefix}_labels.npy"
    np.save(embeddings_path, embeddings)
    np.save(labels_path, labels)
    mapping_path = save_label_mapping(output_dir / f"{output_prefix}_label_mapping.json", class_labels)

    print(f"Embeddings saved: {embeddings_path} shape={embeddings.shape}")
    print(f"Labels saved: {labels_path} shape={labels.shape}")
    print(f"Label mapping saved: {mapping_path}")
    return embeddings_path, labels_path


def load_jsonl_split(data_dir: str | Path, split: str, input_column: str = "func", label_column: str = "target"):
    file_path = Path(data_dir) / f"{split}.jsonl"
    records = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    df = pd.DataFrame(records)
    df = df.dropna(subset=[input_column, label_column]).reset_index(drop=True)
    df[input_column] = df[input_column].astype(str)
    return df


def embed_code_jsonl_splits(
    data_dir: str | Path,
    output_dir: str | Path,
    model_name: str = DEFAULT_CODE_MODEL,
    splits: tuple[str, ...] = ("train", "valid", "test"),
    input_column: str = "func",
    label_column: str = "target",
    max_length: int = 512,
    batch_size: int = 16,
    output_prefix: str = "codebert",
    device: str | torch.device | None = None,
):
    device = get_device(str(device) if device else None)
    data_dir = Path(data_dir)
    output_dir = ensure_dir(output_dir)

    datasets = {
        split: load_jsonl_split(data_dir, split, input_column=input_column, label_column=label_column)
        for split in splits
    }

    for split, df in datasets.items():
        print(f"{split}: {len(df)} samples, labels={df[label_column].value_counts().to_dict()}")

    all_labels = np.concatenate([df[label_column].to_numpy() for df in datasets.values()])
    _, class_labels = encode_labels(all_labels)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    def embed_code_batch(batch_inputs):
        inputs = tokenizer(
            batch_inputs,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = _embedding_from_outputs(outputs, attention_mask=inputs.get("attention_mask"))
        return embeddings.cpu().numpy()

    def embed_split(df, split):
        values = df[input_column].tolist()
        labels, _ = encode_labels(df[label_column].to_numpy(), class_labels=class_labels)
        embeddings = []

        print(f"Processing {len(values)} code samples for split {split}...")
        for i in tqdm(range(0, len(values), batch_size), desc=split):
            embeddings.append(embed_code_batch(values[i : i + batch_size]))

        return np.vstack(embeddings), labels

    manifest = {
        "model_name": model_name,
        "input_column": input_column,
        "label_column": label_column,
        "label_mapping_file": f"{output_prefix}_label_mapping.json",
        "max_length": max_length,
        "batch_size": batch_size,
        "splits": {},
    }

    for split, df in datasets.items():
        embeddings, labels = embed_split(df, split)

        embedding_path = output_dir / f"{output_prefix}_{split}_embeddings.npy"
        label_path = output_dir / f"{output_prefix}_{split}_labels.npy"
        metadata_path = output_dir / f"{output_prefix}_{split}_metadata.csv"

        np.save(embedding_path, embeddings)
        np.save(label_path, labels)

        metadata_columns = [col for col in ["idx", "project", "commit_id"] if col in df.columns]
        if metadata_columns:
            df[metadata_columns].to_csv(metadata_path, index=False)

        manifest["splits"][split] = {
            "samples": int(len(df)),
            "embedding_shape": list(embeddings.shape),
            "embeddings_file": embedding_path.name,
            "labels_file": label_path.name,
            "metadata_file": metadata_path.name if metadata_columns else None,
        }

        print(f"{split} embeddings saved: {embedding_path} shape={embeddings.shape}")
        print(f"{split} labels saved: {label_path} shape={labels.shape}")

    manifest_path = output_dir / f"{output_prefix}_embedding_manifest.json"
    mapping_path = save_label_mapping(output_dir / f"{output_prefix}_label_mapping.json", class_labels)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest saved: {manifest_path}")
    print(f"Label mapping saved: {mapping_path}")
    return manifest


def embed_code_jsonl_split(
    data_dir: str | Path,
    output_dir: str | Path,
    split: str = "train",
    model_name: str = DEFAULT_CODE_MODEL,
    input_column: str = "func",
    label_column: str = "target",
    max_length: int = 512,
    batch_size: int = 16,
    output_prefix: str = "codebert",
    device: str | torch.device | None = None,
):
    device = get_device(str(device) if device else None)
    output_dir = ensure_dir(output_dir)
    df = load_jsonl_split(data_dir, split, input_column=input_column, label_column=label_column)

    print(f"{split}: {len(df)} samples, labels={df[label_column].value_counts().to_dict()}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    def embed_code_batch(batch_inputs):
        inputs = tokenizer(
            batch_inputs,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = _embedding_from_outputs(outputs, attention_mask=inputs.get("attention_mask"))
        return embeddings.cpu().numpy()

    values = df[input_column].tolist()
    labels, class_labels = encode_labels(df[label_column].to_numpy())
    embeddings = []
    print(f"Processing {len(values)} code samples for split {split}...")
    for i in tqdm(range(0, len(values), batch_size), desc=split):
        embeddings.append(embed_code_batch(values[i : i + batch_size]))

    embeddings = np.vstack(embeddings)
    embeddings_path = output_dir / f"{output_prefix}_{split}_embeddings.npy"
    labels_path = output_dir / f"{output_prefix}_{split}_labels.npy"
    metadata_path = output_dir / f"{output_prefix}_{split}_metadata.csv"

    np.save(embeddings_path, embeddings)
    np.save(labels_path, labels)
    mapping_path = save_label_mapping(output_dir / f"{output_prefix}_{split}_label_mapping.json", class_labels)

    metadata_columns = [col for col in ["idx", "project", "commit_id"] if col in df.columns]
    if metadata_columns:
        df[metadata_columns].to_csv(metadata_path, index=False)

    print(f"{split} embeddings saved: {embeddings_path} shape={embeddings.shape}")
    print(f"{split} labels saved: {labels_path} shape={labels.shape}")
    print(f"{split} label mapping saved: {mapping_path}")
    return embeddings_path, labels_path
