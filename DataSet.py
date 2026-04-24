import argparse

from eyetrack.data.openeds import OpenEDSSegDataset, read_gray_image, read_label_npy, read_mask_png
from eyetrack.paths import resolve_openeds_root

__all__ = [
    "OpenEDSSegDataset",
    "read_gray_image",
    "read_label_npy",
    "read_mask_png",
]


def parse_args() -> argparse.Namespace:
    """
    summary: parse dataset inspection arguments
    param none: none
    return: arguments
    """
    parser = argparse.ArgumentParser(description="Inspect an OpenEDS dataset split")
    parser.add_argument("--root_dir", type=str, default=None, help="OpenEDS dataset root directory")
    parser.add_argument("--split", type=str, default="train", help="dataset split, default train")
    return parser.parse_args()


def main() -> None:
    """
    summary: inspect dataset
    param none: none
    return: none
    """
    args = parse_args()
    dataset_root = resolve_openeds_root(args.root_dir)
    dataset = OpenEDSSegDataset(root_dir=str(dataset_root), split=args.split)
    print(f"{args.split.capitalize()} set sample count: {len(dataset)}")
    print(dataset.__getitem__(0))
    sample = dataset[0]
    print(f"Sample ID: {sample['id']}")
    print(f"Image tensor shape: {sample['image'].shape}, dtype: {sample['image'].dtype}")
    print(f"Label tensor shape: {sample['label'].shape}, dtype: {sample['label'].dtype}")
    print(f"Mask tensor shape: {sample['mask'].shape}, dtype: {sample['mask'].dtype}")


if __name__ == "__main__":
    main()
