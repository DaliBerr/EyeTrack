import argparse
import fnmatch
import re
from pathlib import Path
from typing import Any, List, Optional

from extract_geometry_from_segmentation import process_prediction_directory
from eyetrack.workflows.predict import run_prediction_to_npy


def natural_key(text: str) -> List[Any]:
    """
    summary: 生成自然排序键
    param text: 输入字符串
    return: 可用于排序的键列表
    """
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def list_sequence_dirs(sequence_root: str, sequence_glob: str) -> List[Path]:
    """
    summary: 列出所有需要处理的序列目录
    param sequence_root: 序列根目录
    param sequence_glob: 序列匹配模式
    return: 排序后的序列目录列表
    """
    root_path = Path(sequence_root)

    if not root_path.exists():
        raise FileNotFoundError(f"找不到序列根目录: {root_path}")

    sequence_dirs = []
    for path in root_path.iterdir():
        if path.is_dir() and fnmatch.fnmatch(path.name, sequence_glob):
            sequence_dirs.append(path)

    sequence_dirs.sort(key=lambda p: natural_key(p.name))
    return sequence_dirs


def resolve_mask_dir(mask_root: Optional[str], sequence_name: str) -> Optional[str]:
    """
    summary: 解析当前序列对应的有效区域目录
    param mask_root: mask 根目录
    param sequence_name: 当前序列名
    return: 若存在则返回 mask 目录，否则返回 None
    """
    if mask_root is None:
        return None

    candidate = Path(mask_root) / sequence_name
    if candidate.exists() and candidate.is_dir():
        return str(candidate)

    return None


def process_all_sequences(
    sequence_root: str,
    checkpoint_path: str,
    output_root: str,
    sequence_glob: str = "S_*",
    mask_root: Optional[str] = None,
    batch_size: int = 1,
    num_workers: int = 0,
    in_channels: int = 1,
    num_classes: int = 4,
    base_channels: int = 32,
    device: str = "auto",
    iris_class_id: int = 2,
    pupil_class_id: int = 3,
    kernel_size: int = 3,
    iris_min_area: int = 100,
    pupil_min_area: int = 20,
    save_overlay: bool = False,
    save_clean_masks: bool = False,
    skip_existing: bool = False,
) -> None:
    """
    summary: 批量对所有 S 序列执行预测并提取几何参数
    param sequence_root: S 序列根目录
    param checkpoint_path: 模型 checkpoint 路径
    param output_root: 批量输出根目录
    param sequence_glob: 序列匹配模式
    param mask_root: 可选 mask 根目录
    param batch_size: 推理批大小
    param num_workers: DataLoader 进程数
    param in_channels: 模型输入通道数
    param num_classes: 模型输出类别数
    param base_channels: U-Net 基础通道数
    param device: 运行设备
    param iris_class_id: iris 类别编号
    param pupil_class_id: pupil 类别编号
    param kernel_size: 形态学核大小
    param iris_min_area: iris 最小面积阈值
    param pupil_min_area: pupil 最小面积阈值
    param save_overlay: 是否保存叠加图
    param save_clean_masks: 是否保存清理后二值图
    param skip_existing: 若结果已存在是否跳过
    return: 无
    """
    sequence_dirs = list_sequence_dirs(sequence_root=sequence_root, sequence_glob=sequence_glob)

    if len(sequence_dirs) == 0:
        raise RuntimeError(f"没有找到匹配的序列目录: root={sequence_root}, glob={sequence_glob}")

    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    print(f"共找到 {len(sequence_dirs)} 个序列待处理。")

    for index, sequence_dir in enumerate(sequence_dirs, start=1):
        sequence_name = sequence_dir.name
        sequence_output_dir = output_root_path / sequence_name
        pred_dir = sequence_output_dir / "predictions"
        output_csv = sequence_output_dir / "geometry.csv"
        overlay_dir = sequence_output_dir / "overlay" if save_overlay else None
        clean_mask_dir = sequence_output_dir / "clean_masks" if save_clean_masks else None
        mask_dir = resolve_mask_dir(mask_root=mask_root, sequence_name=sequence_name)

        if skip_existing and output_csv.exists():
            print(f"\n[{index}/{len(sequence_dirs)}] 跳过 {sequence_name}，已存在: {output_csv}")
            continue

        print(f"\n[{index}/{len(sequence_dirs)}] 开始处理序列: {sequence_name}")
        print(f"image_dir: {sequence_dir}")
        print(f"pred_dir : {pred_dir}")
        print(f"csv_path : {output_csv}")
        if mask_dir is not None:
            print(f"mask_dir : {mask_dir}")

        run_prediction_to_npy(
            checkpoint_path=checkpoint_path,
            output_dir=str(pred_dir),
            image_dir=str(sequence_dir),
            root_dir=None,
            split="validation",
            batch_size=batch_size,
            num_workers=num_workers,
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            device=device,
        )

        process_prediction_directory(
            pred_dir=str(pred_dir),
            output_csv=str(output_csv),
            image_dir=str(sequence_dir),
            mask_dir=mask_dir,
            overlay_dir=str(overlay_dir) if overlay_dir is not None else None,
            clean_mask_dir=str(clean_mask_dir) if clean_mask_dir is not None else None,
            iris_class_id=iris_class_id,
            pupil_class_id=pupil_class_id,
            kernel_size=kernel_size,
            iris_min_area=iris_min_area,
            pupil_min_area=pupil_min_area,
            save_overlay=save_overlay,
            save_clean_masks=save_clean_masks,
        )

    print(f"\n全部序列处理完成，输出根目录: {output_root_path}")


def parse_args() -> argparse.Namespace:
    """
    summary: 解析命令行参数
    param 无: 无
    return: 参数对象
    """
    parser = argparse.ArgumentParser(description="批量对所有 S 序列执行分割预测并提取 geometry")

    parser.add_argument("--sequence_root", type=str, required=True, help="包含 S_0、S_1 等序列目录的根目录")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument("--output_root", type=str, required=True, help="批量输出根目录")

    parser.add_argument("--sequence_glob", type=str, default="S_*", help="序列匹配模式，默认 S_*")
    parser.add_argument("--mask_root", type=str, default=None, help="可选 mask 根目录，若提供则使用 mask_root/序列名")
    parser.add_argument("--skip_existing", action="store_true", help="若 geometry.csv 已存在则跳过该序列")

    parser.add_argument("--batch_size", type=int, default=1, help="推理批大小")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader 进程数")
    parser.add_argument("--in_channels", type=int, default=1, help="模型输入通道数")
    parser.add_argument("--num_classes", type=int, default=4, help="模型输出类别数")
    parser.add_argument("--base_channels", type=int, default=32, help="U-Net 基础通道数")
    parser.add_argument("--device", type=str, default="auto", help="运行设备: auto/cpu/cuda")

    parser.add_argument("--iris_class_id", type=int, default=2, help="iris 类别编号")
    parser.add_argument("--pupil_class_id", type=int, default=3, help="pupil 类别编号")
    parser.add_argument("--kernel_size", type=int, default=3, help="形态学核大小")
    parser.add_argument("--iris_min_area", type=int, default=100, help="iris 最小面积阈值")
    parser.add_argument("--pupil_min_area", type=int, default=20, help="pupil 最小面积阈值")

    parser.add_argument("--save_overlay", action="store_true", help="是否保存叠加图")
    parser.add_argument("--save_clean_masks", action="store_true", help="是否保存清理后二值图")

    return parser.parse_args()


def main() -> None:
    """
    summary: 主函数，批量执行预测和 geometry 提取
    param 无: 无
    return: 无
    """
    args = parse_args()

    process_all_sequences(
        sequence_root=args.sequence_root,
        checkpoint_path=args.checkpoint_path,
        output_root=args.output_root,
        sequence_glob=args.sequence_glob,
        mask_root=args.mask_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        base_channels=args.base_channels,
        device=args.device,
        iris_class_id=args.iris_class_id,
        pupil_class_id=args.pupil_class_id,
        kernel_size=args.kernel_size,
        iris_min_area=args.iris_min_area,
        pupil_min_area=args.pupil_min_area,
        save_overlay=args.save_overlay,
        save_clean_masks=args.save_clean_masks,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
