from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch


@dataclass
class DetectionState:
    train_df: pd.DataFrame
    clean_labels: np.ndarray
    corrupted_indices: np.ndarray | None
    num_labels: int
    device: torch.device
    detected_noisy_indices: np.ndarray | pd.Index | None = None
    test_forget_rate: float = 0.0
