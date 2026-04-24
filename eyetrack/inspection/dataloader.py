import argparse
from typing import Dict

import torch
from torch.utils.data import DataLoader

from eyetrack.data.openeds import OpenEDSSegDataset
from eyetrack.paths import resolve_openeds_root


def inspect_batch(batch: Dict[str, torch.Tensor]) -> None:
    """
    summary: batch
    param batch: DataLoader return batch
    return: none
    """
    print("image shape:", batch["image"].shape)
    print("label shape:", batch["label"].shape)
    print("mask shape :", batch["mask"].shape)

    print("image dtype:", batch["image"].dtype)
    print("label dtype:", batch["label"].dtype)
    print("mask dtype :", batch["mask"].dtype)

    print("label unique:", torch.unique(batch["label"]))
    print("mask unique :", torch.unique(batch["mask"]))


def parse_args() -> argparse.Namespace:
    """
    summary: parse dataloader inspection arguments
    param none: none
    return: arguments
    """
    parser = argparse.ArgumentParser(description="Inspect one OpenEDS DataLoader batch")
    parser.add_argument("--root_dir", type=str, default=None, help="OpenEDS dataset root directory")
    parser.add_argument("--split", type=str, default="train", help="dataset split, default train")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = resolve_openeds_root(args.root_dir)

    train_dataset = OpenEDSSegDataset(root_dir=str(root_dir), split=args.split)
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    batch = next(iter(train_loader))
    inspect_batch(batch)
    print("sample ids:", batch["id"])


if __name__ == "__main__":
    main()
