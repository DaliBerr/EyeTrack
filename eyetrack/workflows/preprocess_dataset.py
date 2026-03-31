import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from eyetrack.config import DEFAULT_INPUT_HEIGHT, DEFAULT_INPUT_WIDTH
from eyetrack.data.openeds import read_gray_image, read_label_npy, read_mask_png
from eyetrack.data.preprocessing import preprocess_gray_image, resize_binary_mask, resize_label_map


VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def ensure_dir(path: Path) -> None:
    """
    summary: 确保目录存在
    param path: 目录路径
    return: 无
    """
    path.mkdir(parents=True, exist_ok=True)


def natural_key(text: str) -> list[object]:
    """
    summary: 生成自然排序键
    param text: 输入字符串
    return: 排序键列表
    """
    import re

    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def parse_splits(text: str) -> list[str]:
    """
    summary: 解析逗号分隔的 split 列表
    param text: 逗号分隔字符串
    return: split 名称列表
    """
    splits = [item.strip() for item in text.split(",") if item.strip()]
    if len(splits) == 0:
        raise ValueError("至少需要提供一个 split。")
    return splits


def iter_files(folder: Path, suffixes: Iterable[str]) -> list[Path]:
    """
    summary: 列出目录中指定后缀的文件并做自然排序
    param folder: 输入目录
    param suffixes: 允许的后缀集合
    return: 文件路径列表
    """
    items = [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in suffixes]
    items.sort(key=lambda path: natural_key(path.name))
    return items


def save_gray_image(image: np.ndarray, save_path: Path) -> None:
    """
    summary: 保存 uint8 灰度图
    param image: 灰度图
    param save_path: 输出路径
    return: 无
    """
    image_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(image_uint8, mode="L").save(save_path)


def save_binary_mask(mask: np.ndarray, save_path: Path) -> None:
    """
    summary: 保存二值 mask 到 png
    param mask: 二值 mask
    param save_path: 输出路径
    return: 无
    """
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    Image.fromarray(mask_uint8, mode="L").save(save_path)


def preprocess_image_dir(
    source_dir: Path,
    target_dir: Path,
    input_width: int,
    input_height: int,
    skip_existing: bool,
) -> int:
    """
    summary: 预处理 images 目录
    param source_dir: 输入 images 目录
    param target_dir: 输出 images 目录
    param input_width: 目标宽度
    param input_height: 目标高度
    param skip_existing: 是否跳过已存在文件
    return: 处理数量
    """
    ensure_dir(target_dir)
    count = 0

    for src_path in iter_files(source_dir, VALID_IMAGE_EXTS):
        dst_path = target_dir / f"{src_path.stem}.png"
        if skip_existing and dst_path.exists():
            continue

        image = read_gray_image(str(src_path))
        image = preprocess_gray_image(image=image, input_width=input_width, input_height=input_height)
        save_gray_image(image, dst_path)
        count += 1

    return count


def preprocess_label_dir(
    source_dir: Path,
    target_dir: Path,
    input_width: int,
    input_height: int,
    skip_existing: bool,
) -> int:
    """
    summary: 预处理 labels 目录
    param source_dir: 输入 labels 目录
    param target_dir: 输出 labels 目录
    param input_width: 目标宽度
    param input_height: 目标高度
    param skip_existing: 是否跳过已存在文件
    return: 处理数量
    """
    ensure_dir(target_dir)
    count = 0

    for src_path in iter_files(source_dir, {".npy"}):
        dst_path = target_dir / src_path.name
        if skip_existing and dst_path.exists():
            continue

        label = read_label_npy(str(src_path))
        label = resize_label_map(label=label, input_width=input_width, input_height=input_height)
        np.save(dst_path, label)
        count += 1

    return count


def preprocess_mask_dir(
    source_dir: Path,
    target_dir: Path,
    input_width: int,
    input_height: int,
    skip_existing: bool,
) -> int:
    """
    summary: 预处理 masks 目录
    param source_dir: 输入 masks 目录
    param target_dir: 输出 masks 目录
    param input_width: 目标宽度
    param input_height: 目标高度
    param skip_existing: 是否跳过已存在文件
    return: 处理数量
    """
    ensure_dir(target_dir)
    count = 0

    for src_path in iter_files(source_dir, VALID_IMAGE_EXTS):
        dst_path = target_dir / f"{src_path.stem}.png"
        if skip_existing and dst_path.exists():
            continue

        mask = read_mask_png(str(src_path))
        mask = resize_binary_mask(mask=mask, input_width=input_width, input_height=input_height)
        save_binary_mask(mask, dst_path)
        count += 1

    return count


def preprocess_openeds_dataset(
    input_root: str,
    output_root: str,
    splits: list[str],
    input_width: int = DEFAULT_INPUT_WIDTH,
    input_height: int = DEFAULT_INPUT_HEIGHT,
    skip_existing: bool = True,
    process_mask: bool = True,
) -> None:
    """
    summary: 将 OpenEDS 风格数据集离线缩放到固定尺寸
    param input_root: 原始数据集根目录
    param output_root: 输出数据集根目录
    param splits: 待处理 split 列表
    param input_width: 目标宽度
    param input_height: 目标高度
    param skip_existing: 是否跳过已存在文件
    param process_mask: 是否处理 mask 目录
    return: 无
    """
    source_root = Path(input_root)
    target_root = Path(output_root)

    if not source_root.exists():
        raise FileNotFoundError(f"找不到输入数据集根目录: {source_root}")

    ensure_dir(target_root)

    for split in splits:
        split_source_dir = source_root / split
        split_target_dir = target_root / split

        if not split_source_dir.exists():
            raise FileNotFoundError(f"找不到 split 目录: {split_source_dir}")

        image_source_dir = split_source_dir / "images"
        label_source_dir = split_source_dir / "labels"
        mask_source_dir = split_source_dir / "masks"

        if not image_source_dir.exists():
            raise FileNotFoundError(f"找不到 images 目录: {image_source_dir}")

        print(f"\n处理 split: {split}")
        print(f"source: {split_source_dir}")
        print(f"target: {split_target_dir}")

        image_count = preprocess_image_dir(
            source_dir=image_source_dir,
            target_dir=split_target_dir / "images",
            input_width=input_width,
            input_height=input_height,
            skip_existing=skip_existing,
        )
        print(f"images 已处理: {image_count}")

        if label_source_dir.exists():
            label_count = preprocess_label_dir(
                source_dir=label_source_dir,
                target_dir=split_target_dir / "labels",
                input_width=input_width,
                input_height=input_height,
                skip_existing=skip_existing,
            )
            print(f"labels 已处理: {label_count}")

        if process_mask and mask_source_dir.exists():
            mask_count = preprocess_mask_dir(
                source_dir=mask_source_dir,
                target_dir=split_target_dir / "masks",
                input_width=input_width,
                input_height=input_height,
                skip_existing=skip_existing,
            )
            print(f"masks 已处理: {mask_count}")
        elif process_mask:
            print("masks 目录不存在，已跳过。")
        else:
            print("按配置跳过 masks 处理。")

    print(f"\n完成。预处理数据集已输出到: {target_root}")


def parse_args() -> argparse.Namespace:
    """
    summary: 解析命令行参数
    param 无: 无
    return: 参数对象
    """
    parser = argparse.ArgumentParser(description="将 OpenEDS 风格数据集离线缩放到固定尺寸")
    parser.add_argument("--input_root", type=str, required=True, help="原始数据集根目录")
    parser.add_argument("--output_root", type=str, required=True, help="输出数据集根目录")
    parser.add_argument("--splits", type=str, default="train,validation", help="待处理 split，逗号分隔")
    parser.add_argument("--input_width", type=int, default=DEFAULT_INPUT_WIDTH, help="目标宽度")
    parser.add_argument("--input_height", type=int, default=DEFAULT_INPUT_HEIGHT, help="目标高度")
    parser.add_argument("--process_mask", action="store_true", default=True, help="处理 masks 目录")
    parser.add_argument("--no-process_mask", action="store_false", dest="process_mask", help="跳过 masks 目录")
    parser.add_argument("--overwrite_existing", action="store_true", help="若输出已存在则覆盖重算")
    return parser.parse_args()


def main() -> None:
    """
    summary: 主函数
    param 无: 无
    return: 无
    """
    args = parse_args()
    preprocess_openeds_dataset(
        input_root=args.input_root,
        output_root=args.output_root,
        splits=parse_splits(args.splits),
        input_width=args.input_width,
        input_height=args.input_height,
        skip_existing=not args.overwrite_existing,
        process_mask=args.process_mask,
    )


if __name__ == "__main__":
    main()
