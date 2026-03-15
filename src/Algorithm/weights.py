import json
from pathlib import Path

import numpy as np


def _ensure_parent_directory(file_path: str) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def save_weights(
    file_path: str,
    weights: list[np.ndarray],
    biases: list[np.ndarray],
    metadata: None | dict = None,
) -> None:
    payload = {
        "weight_count": len(weights),
        "bias_count": len(biases),
    }
    if metadata:
        payload.update(metadata)

    arrays: dict[str, np.ndarray] = {
        "metadata_json": np.array(json.dumps(payload)),
    }

    for i, weight in enumerate(weights):
        arrays[f"weight_{i}"] = np.asarray(weight)

    for i, bias in enumerate(biases):
        arrays[f"bias_{i}"] = np.asarray(bias)

    _ensure_parent_directory(file_path)
    np.savez(file_path, **arrays)


def load_weights(file_path: str) -> tuple[list[np.ndarray], list[np.ndarray], dict]:
    with np.load(file_path, allow_pickle=False) as data:
        metadata = json.loads(data["metadata_json"].item())

        weight_count = int(metadata["weight_count"])
        bias_count = int(metadata["bias_count"])

        weights = [data[f"weight_{i}"].copy() for i in range(weight_count)]
        biases = [data[f"bias_{i}"].copy() for i in range(bias_count)]

    return weights, biases, metadata