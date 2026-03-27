from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_data_by_suffix(file_path: str) -> Any:
    """
    summary: 根据文件后缀自动选择合适的读取方式
    param file_path: 文件路径，支持 png/jpg/jpeg/bmp/tif/tiff/npy
    return: 读取后的对象，通常为 numpy 数组
    """
    suffix = Path(file_path).suffix.lower()

    if suffix in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        return np.array(Image.open(file_path))

    if suffix == ".npy":
        return np.load(file_path, allow_pickle=True)

    raise ValueError(f"不支持的文件类型: {suffix} -> {file_path}")


def print_basic_info(name: str, data: Any, file_path: str) -> None:
    """
    summary: 打印数据的基础信息
    param name: 数据名称，如 image/label/mask
    param data: 已加载的数据对象
    param file_path: 原始文件路径
    return: 无
    """
    print(f"\n===== {name} =====")
    print(f"[路径] {file_path}")
    print(f"[类型] {type(data)}")

    if isinstance(data, np.ndarray):
        print(f"[shape] {data.shape}")
        print(f"[dtype] {data.dtype}")

        if data.size > 0 and np.issubdtype(data.dtype, np.number):
            print(f"[min] {data.min()}")
            print(f"[max] {data.max()}")

            unique_vals = np.unique(data)
            print(f"[unique前20个] {unique_vals[:20]}")
            print(f"[unique总数] {len(unique_vals)}")
            print(f"[非零像素数] {np.count_nonzero(data)}")
        else:
            print("[提示] 该数组不是数值型，无法统计 min/max/unique。")
    else:
        print("[内容预览]")
        print(data)


def normalize_for_display(data: np.ndarray) -> np.ndarray:
    """
    summary: 将数据处理为适合 matplotlib 显示的形式
    param data: 输入数组
    return: 可显示数组
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
    summary: 将 image、label、mask 并排显示
    param image_data: 图像数据
    param label_data: 标签数据
    param mask_data: 掩码数据
    return: 无
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
    summary: 检查一组 image/label/mask 样本并可视化
    param image_path: 图像路径
    param label_path: 标签路径
    param mask_path: 掩码路径
    return: 无
    """
    image_data = load_data_by_suffix(image_path)
    label_data = load_data_by_suffix(label_path)
    mask_data = load_data_by_suffix(mask_path)

    print_basic_info("image", image_data, image_path)
    print_basic_info("label", label_data, label_path)
    print_basic_info("mask", mask_data, mask_path)

    show_triplet(image_data, label_data, mask_data)


def main() -> None:
    image_path = r"D:\Code\DataSet\OpenEDS\openEDS\openEDS\train\images\000000.png"
    label_path = r"D:\Code\DataSet\OpenEDS\openEDS\openEDS\train\labels\000000.npy"
    mask_path = r"D:\Code\DataSet\OpenEDS\openEDS\openEDS\train\masks\000000.png"

    inspect_sample(image_path, label_path, mask_path)


if __name__ == "__main__":
    main()
