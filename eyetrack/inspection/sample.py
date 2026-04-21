import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from eyetrack.paths import resolve_openeds_sample_paths


def load_data_by_suffix(file_path: str) -> Any:
    """
    summary: file read
    param file_path: filepath, supports png/jpg/jpeg/bmp/tif/tiff/npy
    return: read, numpy array
    """
    suffix = Path(file_path).suffix.lower()

    if suffix in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        return np.array(Image.open(file_path))

    if suffix == ".npy":
        return np.load(file_path, allow_pickle=True)

    raise ValueError(f"unsupported file: {suffix} -> {file_path}")


def print_basic_info(name: str, data: Any, file_path: str) -> None:
    """
    summary:
    param name:, image/label/mask
    param data:
    param file_path: filepath
    return: none
    """
    print(f"\n===== {name} =====")
    print(f"[path] {file_path}")
    print(f"[ ] {type(data)}")

    if isinstance(data, np.ndarray):
        print(f"[shape] {data.shape}")
        print(f"[dtype] {data.dtype}")

        if data.size > 0 and np.issubdtype(data.dtype, np.number):
            print(f"[min] {data.min()}")
            print(f"[max] {data.max()}")

            unique_vals = np.unique(data)
            print(f"[unique 20 ] {unique_vals[:20]}")
            print(f"[unique ] {len(unique_vals)}")
            print(f"[ ] {np.count_nonzero(data)}")
        else:
            print("[ ] array, unable to min/max/unique.")
    else:
        print("[ ]")
        print(data)


def normalize_for_display(data: np.ndarray) -> np.ndarray:
    """
    summary: process matplotlib
    param data: inputarray
    return: array
    """
    if not isinstance(data, np.ndarray):
        return data

    if data.ndim == 2:
        return data

    if data.ndim == 3:
        if data.shape[2] == 1:
            return data[:, :, 0]
        return data

    return data


def show_triplet(image_data: Any, label_data: Any, mask_data: Any) -> None:
    """
    summary: image, label, mask
    param image_data: image
    param label_data: label
    param mask_data: mask
    return: none
    """
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    img_show = normalize_for_display(image_data)
    if isinstance(img_show, np.ndarray) and img_show.ndim == 2:
        plt.imshow(img_show, cmap="gray")
    else:
        plt.imshow(img_show)
    plt.title("image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    lab_show = normalize_for_display(label_data)
    if isinstance(lab_show, np.ndarray):
        plt.imshow(lab_show, cmap="gray")
        plt.colorbar(fraction=0.046, pad=0.04)
    else:
        plt.text(0.1, 0.5, str(lab_show))
    plt.title("label")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    mask_show = normalize_for_display(mask_data)
    if isinstance(mask_show, np.ndarray):
        plt.imshow(mask_show, cmap="gray")
        plt.colorbar(fraction=0.046, pad=0.04)
    else:
        plt.text(0.1, 0.5, str(mask_show))
    plt.title("mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def inspect_sample(image_path: str, label_path: str, mask_path: str) -> None:
    """
    summary: image/label/mask sample
    param image_path: imagepath
    param label_path: labelpath
    param mask_path: maskpath
    return: none
    """
    image_data = load_data_by_suffix(image_path)
    label_data = load_data_by_suffix(label_path)
    mask_data = load_data_by_suffix(mask_path)

    print_basic_info("image", image_data, image_path)
    print_basic_info("label", label_data, label_path)
    print_basic_info("mask", mask_data, mask_path)

    show_triplet(image_data, label_data, mask_data)


def parse_args() -> argparse.Namespace:
    """
    summary: parse sample inspection arguments
    param none: none
    return: arguments
    """
    parser = argparse.ArgumentParser(description="Inspect one OpenEDS sample")
    parser.add_argument("--root_dir", type=str, default=None, help="OpenEDS dataset root directory")
    parser.add_argument("--split", type=str, default="train", help="dataset split, default train")
    parser.add_argument("--sample_id", type=str, default="000000", help="sample id without suffix")
    return parser.parse_args()


def main() -> None:
    """
    summary: inspect one sample
    param none: none
    return: none
    """
    args = parse_args()
    image_path, label_path, mask_path = resolve_openeds_sample_paths(
        root_dir=args.root_dir,
        split=args.split,
        sample_id=args.sample_id,
    )

    inspect_sample(str(image_path), str(label_path), str(mask_path))


if __name__ == "__main__":
    main()
