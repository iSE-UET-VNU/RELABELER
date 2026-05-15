from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def multiclass_noisify(y: np.ndarray, transition_matrix: np.ndarray, seed: int = 2025) -> np.ndarray:
    assert transition_matrix.shape[0] == transition_matrix.shape[1]
    assert np.max(y) < transition_matrix.shape[0]
    assert (transition_matrix >= 0.0).all()

    new_y = y.copy()
    flipper = np.random.RandomState(seed)

    for idx in np.arange(y.shape[0]):
        label = y[idx]
        flipped = flipper.multinomial(1, transition_matrix[label, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0][0]

    print(np.round(transition_matrix * 100, 2))
    return new_y


def noisify_asymmetric(y_train: np.ndarray, noise: float, nb_classes: int = 10, seed: int = 2025):
    transition_matrix = np.eye(nb_classes)
    actual_noise = 0.0

    if noise > 0.0:
        transition_matrix[0, 0], transition_matrix[0, 1] = 1.0 - noise, noise
        for i in range(1, nb_classes - 1):
            transition_matrix[i, i], transition_matrix[i, i + 1] = 1.0 - noise, noise
        transition_matrix[nb_classes - 1, nb_classes - 1] = 1.0 - noise
        transition_matrix[nb_classes - 1, 0] = noise

        y_train_noisy = multiclass_noisify(y_train, transition_matrix, seed=seed)
        actual_noise = (y_train_noisy != y_train).mean()
        y_train = y_train_noisy

    return y_train, actual_noise


def noisify_symmetric(y_train: np.ndarray, noise: float, nb_classes: int = 10, seed: int = 2025):
    transition_matrix = np.ones((nb_classes, nb_classes))
    actual_noise = 0.0
    transition_matrix = (noise / (nb_classes - 1)) * transition_matrix

    if noise > 0.0:
        transition_matrix[np.arange(nb_classes), np.arange(nb_classes)] = 1.0 - noise
        y_train_noisy = multiclass_noisify(y_train, transition_matrix, seed=seed)
        actual_noise = (y_train_noisy != y_train).mean()
        y_train = y_train_noisy

    return y_train, actual_noise


def noisify_instance(train_data: np.ndarray, train_labels: np.ndarray, noise_rate: float):
    num_class = int(np.max(train_labels) + 1)

    q_candidates = np.random.normal(loc=noise_rate, scale=0.1, size=1_000_000)
    q = [pro for pro in q_candidates if 0 < pro < 1]
    weights = [
        np.random.normal(loc=0, scale=1, size=(train_data.shape[1], num_class))
        for _ in range(num_class)
    ]

    noisy_labels = []
    transition_count = np.zeros((num_class, num_class))
    for i, sample in enumerate(train_data):
        p_all = np.matmul(sample, weights[train_labels[i]])
        p_all[train_labels[i]] = -1_000_000
        p_all = q[i] * F.softmax(torch.tensor(p_all), dim=0).numpy()
        p_all[train_labels[i]] = 1 - q[i]
        noisy_labels.append(np.random.choice(np.arange(num_class), p=p_all))
        transition_count[train_labels[i]][noisy_labels[i]] += 1

    overall_noise_rate = 1 - float(torch.tensor(train_labels).eq(torch.tensor(noisy_labels)).sum()) / len(train_labels)
    transition_count = transition_count / np.sum(transition_count, axis=1, keepdims=True)
    print(np.round(transition_count * 100, 1))
    return np.array(noisy_labels), overall_noise_rate
