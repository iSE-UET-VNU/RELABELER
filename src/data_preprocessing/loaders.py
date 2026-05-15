from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .label_encoding import encode_labels
from .noise import noisify_asymmetric, noisify_instance, noisify_symmetric


def init_data(
    features_path: str,
    labels_path: str,
    noise_config: dict[str, float] | None = None,
    sampling_rate: float = 1,
    num_gt: int = 0,
    seed: int = 2025,
):
    train_df = pd.DataFrame(np.load(features_path))
    raw_labels = np.load(labels_path, allow_pickle=True).flatten()
    clean_labels, _ = encode_labels(raw_labels)
    train_df["labels"] = clean_labels
    data_gt = None

    if not noise_config:
        print("No noise config provided. Returning clean data.")
        return train_df, clean_labels, np.array([]), train_df.shape[0], train_df["labels"].nunique(), None

    if sampling_rate != 1:
        train_df = train_df.sample(frac=sampling_rate, random_state=seed)
        clean_labels = train_df["labels"].values
        train_df.reset_index(drop=True, inplace=True)

    if num_gt != 0:
        train_df, data_gt = train_test_split(train_df, test_size=num_gt, random_state=seed)
        clean_labels = train_df["labels"].values
        train_df.reset_index(drop=True, inplace=True)
        data_gt.reset_index(drop=True, inplace=True)

    num_samples = train_df.shape[0]
    num_labels = train_df["labels"].nunique()
    noisy_labels = clean_labels.copy()
    all_flipped_indices = []
    available_indices = np.arange(num_samples)
    np.random.shuffle(available_indices)
    features = train_df.iloc[:, :-1].values
    print("--------------Generating Mixed Noise (Guaranteed Flip)---------------")

    for noise_type, noise_rate in noise_config.items():
        if noise_rate == 0:
            continue

        num_to_corrupt = int(num_samples * noise_rate)
        if num_to_corrupt == 0 or len(available_indices) < num_to_corrupt:
            print(f"Warning: Not enough samples to apply {noise_rate * 100}% of {noise_type} noise.")
            continue

        indices_to_noise = available_indices[:num_to_corrupt]
        available_indices = available_indices[num_to_corrupt:]
        subset_clean_labels = clean_labels[indices_to_noise]

        print(f"\nFlipping {len(indices_to_noise)} samples using {noise_type} rule...")
        if noise_type == "ins":
            subset_features = features[indices_to_noise]
            subset_noisy_labels, _ = noisify_instance(subset_features, subset_clean_labels, noise_rate=1.0)
        elif noise_type == "sym":
            subset_noisy_labels, _ = noisify_symmetric(
                subset_clean_labels,
                noise=1.0,
                nb_classes=num_labels,
                seed=seed,
            )
        elif noise_type == "asym":
            subset_noisy_labels, _ = noisify_asymmetric(
                subset_clean_labels,
                noise=1.0,
                nb_classes=num_labels,
                seed=seed,
            )
        else:
            print(f"Warning: Unknown noise type '{noise_type}' found in config. Skipping.")
            continue

        noisy_labels[indices_to_noise] = subset_noisy_labels
        flipped_mask = subset_clean_labels != subset_noisy_labels
        all_flipped_indices.extend(np.array(indices_to_noise)[flipped_mask])

    train_df["labels"] = noisy_labels
    corrupted_indices = np.unique(all_flipped_indices)
    actual_noise_rate = len(corrupted_indices) / num_samples

    print("\n--------------Data Info---------------")
    print(f"Total configured noise rate: {sum(noise_config.values()) * 100:.2f}%")
    print(f"Actual final noise rate: {(actual_noise_rate * 100):.2f}%")
    print(f"Num of samples: {num_samples}")
    print(f"Num of noisy samples: {len(corrupted_indices)}")

    return train_df, clean_labels, corrupted_indices, num_samples, num_labels, data_gt
