from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from correction import LabelCorrector
from core.utils import ensure_dir, get_device, seed_everything
from data_preprocessing import embed_code_jsonl_split, embed_csv_images, embed_csv_text, init_data
from detection import DetectionState, global_step, local_step
from evaluation import calculate_correction_metrics


def run_relabeler(
    *,
    raw_csv_path: str | None = None,
    raw_image_csv_path: str | None = None,
    code_jsonl_data_dir: str | None = None,
    text_column: str = "text",
    label_column: str = "label",
    image_path_column: str = "image_path",
    image_label_column: str = "label",
    code_column: str = "func",
    code_label_column: str = "target",
    code_split: str = "train",
    embedding_model_name: str | None = None,
    embedding_batch_size: int = 32,
    embedding_max_length: int = 512,
    noise_config: dict[str, float] | None = None,
    sampling_rate: float = 1,
    num_gt: int = 0,
    seed: int = 2025,
    k_neighbors: int = 30,
    smooth_value: float = 1e-10,
    n_iterations: int = -1,
    correction_confidence_threshold: float = 0.9,
    global_epochs: int = 200,
    output_dir: str | Path = "artifacts/outputs",
    model_dir: str | Path = "artifacts/models",
    device: str | None = None,
):
    output_dir = ensure_dir(output_dir)
    device_obj = get_device(device)
    print("Device:", device_obj)
    seed_everything(seed)

    if raw_csv_path:
        print("=========== GENERATING TEXT EMBEDDINGS FROM RAW CSV ===========\n")
        features_path, labels_path = embed_csv_text(
            csv_file_path=raw_csv_path,
            output_dir=output_dir / "embeddings",
            text_column=text_column,
            label_column=label_column,
            model_name=embedding_model_name or "bert-base-uncased",
            batch_size=embedding_batch_size,
            output_prefix=Path(raw_csv_path).stem,
            max_length=embedding_max_length,
            device=device_obj,
        )
        train_df, clean_labels, corrupted_indices, num_samples, num_labels, data_gt = init_data(
            features_path=str(features_path),
            labels_path=str(labels_path),
            noise_config=noise_config,
            sampling_rate=sampling_rate,
            num_gt=num_gt,
            seed=seed,
        )
    elif raw_image_csv_path:
        print("=========== GENERATING CLIP EMBEDDINGS FROM RAW IMAGES ===========\n")
        features_path, labels_path = embed_csv_images(
            csv_file_path=raw_image_csv_path,
            output_dir=output_dir / "embeddings",
            model_name=embedding_model_name or "openai/clip-vit-base-patch32",
            image_path_column=image_path_column,
            label_column=image_label_column,
            batch_size=embedding_batch_size,
            output_prefix=Path(raw_image_csv_path).stem,
            device=device_obj,
        )
        train_df, clean_labels, corrupted_indices, num_samples, num_labels, data_gt = init_data(
            features_path=str(features_path),
            labels_path=str(labels_path),
            noise_config=noise_config,
            sampling_rate=sampling_rate,
            num_gt=num_gt,
            seed=seed,
        )
    elif code_jsonl_data_dir:
        print("=========== GENERATING CODEBERT EMBEDDINGS FROM RAW CODE ===========\n")
        features_path, labels_path = embed_code_jsonl_split(
            data_dir=code_jsonl_data_dir,
            output_dir=output_dir / "embeddings",
            split=code_split,
            model_name=embedding_model_name or "microsoft/codebert-base",
            input_column=code_column,
            label_column=code_label_column,
            max_length=embedding_max_length,
            batch_size=embedding_batch_size,
            device=device_obj,
        )
        train_df, clean_labels, corrupted_indices, num_samples, num_labels, data_gt = init_data(
            features_path=str(features_path),
            labels_path=str(labels_path),
            noise_config=noise_config,
            sampling_rate=sampling_rate,
            num_gt=num_gt,
            seed=seed,
        )
    else:
        raise ValueError("Provide raw_csv_path, raw_image_csv_path, or code_jsonl_data_dir.")

    initial_noisy_labels = train_df["labels"].values.copy()
    state = DetectionState(
        train_df=train_df,
        clean_labels=clean_labels,
        corrupted_indices=corrupted_indices,
        num_labels=num_labels,
        device=device_obj,
    )

    print("=========== RUNNING DETECTION PHASE (LOCAL + GLOBAL STEPS) ===========\n")
    local_step(state, k_neighbors=k_neighbors, smooth_value=smooth_value)
    clean_model = global_step(
        state,
        n_iterations=n_iterations,
        num_epochs=global_epochs,
        seed=seed,
    )

    print("\n=========== PREPARE LABEL CORRECTION ===========\n")
    final_corrupted_indices = np.array(state.detected_noisy_indices if state.detected_noisy_indices is not None else [])
    all_indices = set(state.train_df.index)
    final_clean_indices = np.array(list(all_indices - set(final_corrupted_indices)))

    print(
        f"After detection phase: {len(final_clean_indices)} clean samples, "
        f"{len(final_corrupted_indices)} corrupted samples remaining."
    )

    if clean_model is None or len(final_corrupted_indices) == 0:
        print("Skipping correction because no trained detection model or no corrupted samples remain.")
        final_corrected_df = state.train_df.copy()
    else:
        df_corrupted_final = state.train_df.loc[final_corrupted_indices]
        x_corrupted_final = torch.tensor(df_corrupted_final.iloc[:, :-1].values, dtype=torch.float32)

        clean_model.model1.eval()
        clean_model.model2.eval()
        with torch.no_grad():
            logits1 = clean_model.model1(x_corrupted_final.to(device_obj))
            logits2 = clean_model.model2(x_corrupted_final.to(device_obj))
            y_pred_clean_soft = F.softmax((logits1 + logits2) / 2, dim=1).cpu().numpy()

        print("\n=========== RUNNING CORRECTION PHASE ===========\n")
        corrector = LabelCorrector(num_classes=num_labels, device=device_obj, model_dir=model_dir)
        final_corrected_df = corrector.correct_labels(
            train_df=state.train_df,
            clean_indices=final_clean_indices,
            corrupted_indices=final_corrupted_indices,
            y_pred_clean_soft=y_pred_clean_soft,
            confidence_threshold=correction_confidence_threshold,
        )

    print("\n=========== FINAL EVALUATION ===========\n")
    metrics = calculate_correction_metrics(
        original_clean_labels=clean_labels,
        initial_noisy_labels=initial_noisy_labels,
        final_corrected_labels=final_corrected_df["labels"].values,
    )

    corrected_output_path = output_dir / "final_corrected_labels.csv"
    final_corrected_df.to_csv(corrected_output_path, index=False)
    print(f"Final corrected data saved: {corrected_output_path}")

    return {
        "state": state,
        "clean_model": clean_model,
        "final_corrected_df": final_corrected_df,
        "metrics": metrics,
        "num_samples": num_samples,
        "num_labels": num_labels,
        "data_gt": data_gt,
    }
