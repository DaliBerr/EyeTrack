from eyetrack.data.openeds import OpenEDSSegDataset, read_gray_image, read_label_npy, read_mask_png

__all__ = [
    "OpenEDSSegDataset",
    "read_gray_image",
    "read_label_npy",
    "read_mask_png",
]


if __name__ == "__main__":
    dataset_root = r"D:\Code\DataSet\OpenEDS\openEDS\openEDS"
    train_dataset = OpenEDSSegDataset(root_dir=dataset_root, split="train")
    print(f"训练集样本数量: {len(train_dataset)}")
    print(train_dataset.__getitem__(0))
    sample = train_dataset[0]
    print(f"样本 ID: {sample['id']}")
    print(f"图像张量形状: {sample['image'].shape}, 数据类型: {sample['image'].dtype}")
    print(f"标签张量形状: {sample['label'].shape}, 数据类型: {sample['label'].dtype}")
    print(f"掩码张量形状: {sample['mask'].shape}, 数据类型: {sample['mask'].dtype}")
