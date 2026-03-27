from typing import Dict

import torch
from torch.utils.data import DataLoader

from eyetrack.data.openeds import OpenEDSSegDataset


def inspect_batch(batch: Dict[str, torch.Tensor]) -> None:
    """
    summary: 打印一个 batch 的基础信息
    param batch: DataLoader 返回的批次数据
    return: 无
    """
    print("image shape:", batch["image"].shape)
    print("label shape:", batch["label"].shape)
    print("mask shape :", batch["mask"].shape)

    print("image dtype:", batch["image"].dtype)
    print("label dtype:", batch["label"].dtype)
    print("mask dtype :", batch["mask"].dtype)

    print("label unique:", torch.unique(batch["label"]))
    print("mask unique :", torch.unique(batch["mask"]))


def main() -> None:
    root_dir = r"D:\Code\DataSet\OpenEDS\openEDS\openEDS"

    train_dataset = OpenEDSSegDataset(root_dir=root_dir, split="train")
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
