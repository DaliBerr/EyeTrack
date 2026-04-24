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
    summary: directory
    param path: directorypath
    return: none
    """
    path.mkdir(parents=True, exist_ok=True)


def natural_key(text: str) -> list[object]:
    """
    summary:
    param text: input
    return: list
    """
    import re

    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def parse_splits(text: str) -> list[str]:
    """
    summary: parse split list
    param text:
    return: split list
    """
    splits = [item.strip() for item in text.split(",") if item.strip()]
    if len(splits) == 0:
        raise ValueError("At least one split must be provided.")
    return splits


def iter_files(folder: Path, suffixes: Iterable[str]) -> list[Path]:
    """
    summary: directory file
    param folder: inputdirectory
    param suffixes:
    return: filepathlist
    """
    items = [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in suffixes]
    items.sort(key=lambda path: natural_key(path.name))
    return items


def save_gray_image(image: np.ndarray, save_path: Path) -> None:
    """
    summary: save uint8
    param image:
    param save_path: outputpath
    return: none
    """
    image_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(image_uint8, mode="L").save(save_path)


def save_binary_mask(mask: np.ndarray, save_path: Path) -> None:
    """
    summary: save mask png
    param mask: mask
    param save_path: outputpath
    return: none
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
    summary: process images directory
    param source_dir: input images directory
    param target_dir: output images directory
    param input_width:
    param input_height:
    param skip_existing: file
    return: processcount
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
    summary: process labels directory
    param source_dir: input labels directory
    param target_dir: output labels directory
    param input_width:
    param input_height:
    param skip_existing: file
    return: processcount
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
    summary: process masks directory
    param source_dir: input masks directory
    param target_dir: output masks directory
    param input_width:
    param input_height:
    param skip_existing: file
    return: processcount
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
    summary: OpenEDS dataset
    param input_root: dataset directory
    param output_root: outputdataset directory
    param splits: process split list
    param input_width:
    param input_height:
    param skip_existing: file
    param process_mask: process mask directory
    return: none
    """
    source_root = Path(input_root)
    target_root = Path(output_root)

    if not source_root.exists():
        raise FileNotFoundError(f"not foundinputdataset directory: {source_root}")

    ensure_dir(target_root)

    for split in splits:
        split_source_dir = source_root / split
        split_target_dir = target_root / split

        if not split_source_dir.exists():
            raise FileNotFoundError(f"not found split directory: {split_source_dir}")

        image_source_dir = split_source_dir / "images"
        label_source_dir = split_source_dir / "labels"
        mask_source_dir = split_source_dir / "masks"

        if not image_source_dir.exists():
            raise FileNotFoundError(f"not found images directory: {image_source_dir}")

        print(f"\nprocess split: {split}")
        print(f"source: {split_source_dir}")
        print(f"target: {split_target_dir}")

        image_count = preprocess_image_dir(
            source_dir=image_source_dir,
            target_dir=split_target_dir / "images",
            input_width=input_width,
            input_height=input_height,
            skip_existing=skip_existing,
        )
        print(f"images process: {image_count}")

        if label_source_dir.exists():
            label_count = preprocess_label_dir(
                source_dir=label_source_dir,
                target_dir=split_target_dir / "labels",
                input_width=input_width,
                input_height=input_height,
                skip_existing=skip_existing,
            )
            print(f"labels process: {label_count}")

        if process_mask and mask_source_dir.exists():
            mask_count = preprocess_mask_dir(
                source_dir=mask_source_dir,
                target_dir=split_target_dir / "masks",
                input_width=input_width,
                input_height=input_height,
                skip_existing=skip_existing,
            )
            print(f"masks process: {mask_count}")
        elif process_mask:
            print("masks directory,.")
        else:
            print(" masks process.")

    print(f"\ncompleted. processdataset output: {target_root}")


def parse_args() -> argparse.Namespace:
    """
    summary: parseCLIarguments
    param none: none
    return: arguments
    """
    parser = argparse.ArgumentParser(description=" OpenEDS dataset ")
    parser.add_argument("--input_root", type=str, required=True, help=" dataset directory")
    parser.add_argument("--output_root", type=str, required=True, help="outputdataset directory")
    parser.add_argument("--splits", type=str, default="train,validation", help=" process split, ")
    parser.add_argument("--input_width", type=int, default=DEFAULT_INPUT_WIDTH, help=" ")
    parser.add_argument("--input_height", type=int, default=DEFAULT_INPUT_HEIGHT, help=" ")
    parser.add_argument("--process_mask", action="store_true", default=True, help="process masks directory")
    parser.add_argument("--no-process_mask", action="store_false", dest="process_mask", help=" masks directory")
    parser.add_argument("--overwrite_existing", action="store_true", help=" output ")
    return parser.parse_args()


def main() -> None:
    """
    summary: main function
    param none: none
    return: none
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
