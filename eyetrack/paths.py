from __future__ import annotations

import os
from pathlib import Path


OPENEDS_ROOT_ENV_VAR = "OPENEDS_ROOT_DIR"


def resolve_openeds_root(root_dir: str | None = None) -> Path:
    """
    summary: resolve OpenEDS root directory
    param root_dir: optional CLI path
    return: dataset root path
    """
    if root_dir:
        return Path(root_dir).expanduser()

    env_root = os.getenv(OPENEDS_ROOT_ENV_VAR)
    if env_root:
        return Path(env_root).expanduser()

    raise ValueError(
        "OpenEDS dataset root is not configured. Pass --root_dir or set OPENEDS_ROOT_DIR."
    )


def resolve_openeds_sample_paths(root_dir: str | None, split: str, sample_id: str) -> tuple[Path, Path, Path]:
    """
    summary: resolve one OpenEDS sample path triplet
    param root_dir: optional CLI path
    param split: dataset split
    param sample_id: sample id without suffix
    return: image, label, mask paths
    """
    dataset_root = resolve_openeds_root(root_dir)
    split_root = dataset_root / split

    image_path = split_root / "images" / f"{sample_id}.png"
    label_path = split_root / "labels" / f"{sample_id}.npy"
    mask_path = split_root / "masks" / f"{sample_id}.png"
    return image_path, label_path, mask_path
