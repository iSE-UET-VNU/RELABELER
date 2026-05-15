from __future__ import annotations

import torch


def infer_clean_label(
    transition_matrix_for_sample: torch.Tensor,
    noisy_label: int,
    clean_label_prior: torch.Tensor,
) -> tuple[torch.Tensor, int, float]:
    """Infer the most likely clean label with the Bayes equation from the paper.

    The derivation in ``bayesian_inference.tex`` starts from:

        P(y | x, y_tilde)
        = P(y_tilde | y, x) * P(y | x) / P(y_tilde | x)

    In this function:
    - ``transition_matrix_for_sample[:, noisy_label]`` is P(y_tilde | y, x)
      for every candidate clean label y.
    - ``clean_label_prior`` is P(y | x), estimated by the detection model.
    - ``posterior.sum()`` is P(y_tilde | x), the evidence term obtained by
      marginalizing over all candidate clean labels.
    """
    p_y_noise_given_y_clean_x = transition_matrix_for_sample[:, int(noisy_label)]

    # Numerator from the derivation:
    # P(y_tilde | y, x) * P(y | x)
    posterior = p_y_noise_given_y_clean_x * clean_label_prior

    # Evidence term:
    # P(y_tilde | x) = sum_k P(y_tilde | y=k, x) * P(y=k | x)
    if posterior.sum() > 0:
        posterior = posterior / posterior.sum()

    # y* = argmax_y P(y | x, y_tilde)
    max_prob = torch.max(posterior).item()
    new_label = torch.argmax(posterior).item()
    return posterior, new_label, max_prob
