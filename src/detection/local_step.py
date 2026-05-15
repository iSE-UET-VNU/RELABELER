from __future__ import annotations

import numpy as np
import torch

from detection.state import DetectionState
from evaluation import print_detection_results


def _build_faiss_index(x: np.ndarray):
    import faiss

    index_flat = faiss.IndexFlatL2(x.shape[1])
    if hasattr(faiss, "StandardGpuResources") and torch.cuda.is_available():
        try:
            res = faiss.StandardGpuResources()
            return faiss.index_cpu_to_gpu(res, 0, index_flat)
        except Exception as exc:
            print(f"Warning: FAISS GPU is not available, falling back to CPU. Detail: {exc}")
    return index_flat


def local_step(state: DetectionState, k_neighbors: int, smooth_value: float = 1e-10) -> DetectionState:
    x = np.ascontiguousarray(state.train_df.iloc[:, :-1].values.astype("float32"))
    y = state.train_df.iloc[:, -1].values

    if len(x) < 2:
        raise ValueError("local_step requires at least two samples.")
    k_neighbors = min(k_neighbors, len(x) - 1)

    index = _build_faiss_index(x)
    index.add(x)

    distances, neighbor_indices = index.search(x, k_neighbors + 1)
    neighbor_indices = neighbor_indices[:, 1 : k_neighbors + 1]
    distances = distances[:, 1 : k_neighbors + 1]

    predicted_labels = []
    for i in range(len(neighbor_indices)):
        neighbor_labels = y[neighbor_indices[i]]
        neighbor_distances = distances[i]

        weights = 1 / (neighbor_distances**2 + smooth_value)
        weight_sum = np.sum(weights)
        weighted_votes = np.zeros(state.num_labels)

        for j in range(len(neighbor_labels)):
            weighted_votes[neighbor_labels[j]] += weights[j]

        weighted_prob = weighted_votes / weight_sum
        predicted_labels.append(np.argmax(weighted_prob))

    predicted_labels = np.array(predicted_labels)
    state.detected_noisy_indices = state.train_df[predicted_labels != y].index
    state.test_forget_rate = print_detection_results(
        train_df=state.train_df,
        clean_labels=state.clean_labels,
        corrupted_indices=state.corrupted_indices,
        detected_noisy_indices=state.detected_noisy_indices,
        phase="local",
    )
    return state
