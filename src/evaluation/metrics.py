from __future__ import annotations

import numpy as np
import pandas as pd


def print_detection_results(
    train_df: pd.DataFrame,
    clean_labels: np.ndarray,
    corrupted_indices: np.ndarray | None,
    detected_noisy_indices,
    phase: str = "global",
    iteration: int = 0,
) -> float:
    current_detected = set(list(detected_noisy_indices))
    true_noisy = set(corrupted_indices if corrupted_indices is not None else [])

    if not true_noisy:
        print("Warning: corrupted_indices (ground truth) is empty or None. Cannot calculate recall/F1.")
        recall = f1_score = 0.0
    else:
        correct_detection_indices = current_detected.intersection(true_noisy)
        recall = len(correct_detection_indices) / len(true_noisy) if len(true_noisy) > 0 else 0.0

    if not current_detected:
        print("Warning: detected_noisy_indices is empty. Cannot calculate precision/F1.")
        precision = f1_score = 0.0
    else:
        correct_detection_indices = current_detected.intersection(true_noisy)
        precision = len(correct_detection_indices) / len(current_detected) if len(current_detected) > 0 else 0.0

    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    wrong_detection_indices = current_detected - true_noisy
    correct_detection_indices_list = list(current_detected.intersection(true_noisy))

    print("--------------------------------------------")
    print(f"After Iteration: {iteration + 1}")
    print("\nDetection Info:")
    print("Global Verification:" if phase == "global" else "Local Verification:")
    print(f"Precision: {round(precision, 3)}")
    print(f"Recall: {round(recall, 3)}")
    print(f"F1: {round(f1_score, 3)}")
    print(f"# of wrong detected error instances (false positives): {len(wrong_detection_indices)}")
    print(f"# of true detected error instances (true positives): {len(correct_detection_indices_list)}")
    print(f"# of remaining noisy instances to check next: {len(detected_noisy_indices)}")
    print("--------------------------------------------\n")

    return 1 - recall


def calculate_correction_metrics(
    original_clean_labels: np.ndarray,
    initial_noisy_labels: np.ndarray,
    final_corrected_labels: np.ndarray,
) -> dict[str, float]:
    def calculate_purity(assigned_labels, true_labels):
        num_classes = len(np.unique(true_labels))
        class_purities = []
        df = pd.DataFrame({"assigned": assigned_labels, "true": true_labels})

        for class_label in range(num_classes):
            class_subset = df[df["assigned"] == class_label]
            if class_subset.empty:
                purity = 0.0
            else:
                majority_class_count = class_subset["true"].value_counts().max()
                purity = majority_class_count / len(class_subset)
            class_purities.append(purity)

        return np.mean(class_purities) if class_purities else 0.0

    purity_before = calculate_purity(initial_noisy_labels, original_clean_labels)
    purity_after = calculate_purity(final_corrected_labels, original_clean_labels)
    accuracy_before = np.mean(initial_noisy_labels == original_clean_labels)
    accuracy_after = np.mean(final_corrected_labels == original_clean_labels)

    total_samples = len(original_clean_labels)
    changed_indices = np.where(initial_noisy_labels != final_corrected_labels)[0]
    num_changed = len(changed_indices)

    if num_changed == 0:
        accuracy_correction = 1.0
        correction_rate = 0.0
        num_correctly_fixed = 0
    else:
        num_correctly_fixed = np.sum(final_corrected_labels[changed_indices] == original_clean_labels[changed_indices])
        accuracy_correction = num_correctly_fixed / num_changed
        correction_rate = num_changed / total_samples

    print("\n--- Final Evaluation Metrics ---")
    print(f"Purity of Initial Noisy Dataset:       {purity_before:.2%}")
    print(f"Purity of Final Corrected Dataset:     {purity_after:.2%}")
    print("---")
    print(f"Overall Accuracy of Initial Dataset:   {accuracy_before:.2%}")
    print(f"Overall Accuracy of Final Dataset:     {accuracy_after:.2%}")
    print("\nCorrection Specifics:")
    print(f"Correction Rate (labels changed):    {correction_rate:.2%} ({num_changed}/{total_samples})")
    print(f"Accuracy of Correction (on changed): {accuracy_correction:.2%} ({num_correctly_fixed}/{num_changed})")
    print("---------------------------------------\n")

    return {
        "purity_before": float(purity_before),
        "purity_after": float(purity_after),
        "accuracy_before": float(accuracy_before),
        "accuracy_after": float(accuracy_after),
        "correction_rate": float(correction_rate),
        "accuracy_correction": float(accuracy_correction),
    }
