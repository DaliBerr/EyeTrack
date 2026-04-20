import argparse
import math
import os
import random
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from eyetrack.config import (
    DEFAULT_BASE_CHANNELS,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_IN_CHANNELS,
    DEFAULT_INPUT_HEIGHT,
    DEFAULT_INPUT_WIDTH,
    DEFAULT_NUM_CLASSES,
    DEFAULT_PREPROCESS_MODE,
    DEFAULT_QAT_BACKEND,
    DEFAULT_QUANTIZATION_MODE,
    DEFAULT_USE_AMP,
    DEFAULT_USE_MASK,
    build_model_metadata,
)
from eyetrack.data.openeds import OpenEDSSegDataset
from eyetrack.models.unet import UNet
from eyetrack.training.checkpoints import load_checkpoint_flexible, resolve_model_metadata, save_full_checkpoint
from eyetrack.training.engine import train_one_epoch, validate_one_epoch
from eyetrack.training.qat_engine import prepare_model_for_qat, update_qat_epoch_state
from eyetrack.runtime import resolve_device, should_enable_amp


def set_seed(seed: int = 42) -> None:
    """
    summary: 固定随机种子以保证结果可复现
    param seed: 随机种子
    return: 无
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_metrics(epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
    """
    summary: 打印训练与验证指标
    param epoch: 当前轮数
    param train_metrics: 训练指标
    param val_metrics: 验证指标
    return: 无
    """
    def render_metric(value: float) -> str:
        if math.isnan(value):
            return "n/a"

        return f"{value:.4f}"

    print(f"\nEpoch [{epoch}]")
    print(
        f"Train | loss={train_metrics['loss']:.4f} "
        f"acc={train_metrics['acc']:.4f} "
        f"masked_acc={render_metric(train_metrics['masked_acc'])} "
        f"dice1={train_metrics['dice_1']:.4f} "
        f"dice2={train_metrics['dice_2']:.4f} "
        f"dice3={train_metrics['dice_3']:.4f}"
    )
    print(
        f"Val   | loss={val_metrics['loss']:.4f} "
        f"acc={val_metrics['acc']:.4f} "
        f"masked_acc={render_metric(val_metrics['masked_acc'])} "
        f"dice1={val_metrics['dice_1']:.4f} "
        f"dice2={val_metrics['dice_2']:.4f} "
        f"dice3={val_metrics['dice_3']:.4f}"
    )


def run_training(
    root_dir: str,
    batch_size: int = 4,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    num_workers: int = 0,
    save_checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
    resume_checkpoint_path: str | None = None,
    in_channels: int = DEFAULT_IN_CHANNELS,
    num_classes: int = DEFAULT_NUM_CLASSES,
    base_channels: int = DEFAULT_BASE_CHANNELS,
    input_width: int = DEFAULT_INPUT_WIDTH,
    input_height: int = DEFAULT_INPUT_HEIGHT,
    use_amp: bool = DEFAULT_USE_AMP,
    use_mask: bool = DEFAULT_USE_MASK,
    device: str = "auto",
    qat_mode: str = "off",
    qat_backend: str = DEFAULT_QAT_BACKEND,
    qat_learning_rate: float | None = None,
    qat_disable_observer_last_n_epochs: int = 1,
    qat_freeze_bn_after_epoch: int = 2,
) -> None:
    """
    summary: 训练 OpenEDS 分割模型并支持 AMP 与轻量配置
    param root_dir: 数据集根目录
    param batch_size: 批大小
    param num_epochs: 训练轮数
    param learning_rate: 学习率
    param num_workers: DataLoader 进程数
    param save_checkpoint_path: checkpoint 保存路径
    param resume_checkpoint_path: 可选续训 checkpoint
    param in_channels: 输入通道数
    param num_classes: 类别数量
    param base_channels: U-Net 基础通道数
    param input_width: 模型输入宽度
    param input_height: 模型输入高度
    param use_amp: 是否启用 AMP
    param use_mask: 是否读取 mask 并计算 masked_acc
    param device: 运行设备
    param qat_mode: QAT 模式，off 或 fine_tune
    param qat_backend: QAT backend，qnnpack 或 fbgemm
    param qat_learning_rate: QAT 阶段可选学习率覆盖
    param qat_disable_observer_last_n_epochs: QAT 最后多少轮关闭 observer
    param qat_freeze_bn_after_epoch: QAT 从第几轮开始冻结 BN 统计
    return: 无
    """
    set_seed(42)

    torch_device = resolve_device(device)
    is_qat_mode = qat_mode == "fine_tune"
    resume_metadata_path = resume_checkpoint_path if resume_checkpoint_path and os.path.exists(resume_checkpoint_path) else None
    resolved_model_metadata = resolve_model_metadata(
        checkpoint_path=resume_metadata_path,
        device="cpu",
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        input_width=input_width,
        input_height=input_height,
        preprocess_mode=DEFAULT_PREPROCESS_MODE,
        amp=use_amp,
        use_mask=use_mask,
        quantization_mode=DEFAULT_QUANTIZATION_MODE,
        qat_backend=qat_backend,
    )

    in_channels = int(resolved_model_metadata["in_channels"])
    num_classes = int(resolved_model_metadata["num_classes"])
    base_channels = int(resolved_model_metadata["base_channels"])
    input_width = int(resolved_model_metadata["input_width"])
    input_height = int(resolved_model_metadata["input_height"])
    use_amp = bool(resolved_model_metadata["amp"])
    use_mask = bool(resolved_model_metadata["use_mask"])
    checkpoint_quantization_mode = str(resolved_model_metadata["quantization_mode"])
    checkpoint_qat_backend = str(resolved_model_metadata["qat_backend"])

    if is_qat_mode and use_amp:
        print("QAT 模式下将自动禁用 AMP 以保证量化统计稳定性。")
        use_amp = False

    selected_learning_rate = learning_rate
    if is_qat_mode and qat_learning_rate is not None:
        selected_learning_rate = qat_learning_rate

    effective_qat_backend = qat_backend
    if is_qat_mode and checkpoint_qat_backend in {"qnnpack", "fbgemm"}:
        effective_qat_backend = checkpoint_qat_backend

    amp_enabled = should_enable_amp(device=torch_device, use_amp=use_amp)
    print("device:", torch_device)
    print("amp:", amp_enabled)
    if is_qat_mode:
        print("qat_mode:", qat_mode)
        print("qat_backend:", effective_qat_backend)
        print("qat_lr:", selected_learning_rate)
    print("model metadata:", resolved_model_metadata)

    train_dataset = OpenEDSSegDataset(
        root_dir=root_dir,
        split="train",
        input_width=input_width,
        input_height=input_height,
        use_mask=use_mask,
    )
    val_dataset = OpenEDSSegDataset(
        root_dir=root_dir,
        split="validation",
        input_width=input_width,
        input_height=input_height,
        use_mask=use_mask,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch_device.type == "cuda",
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch_device.type == "cuda",
    )

    model = UNet(in_channels=in_channels, num_classes=num_classes, base_channels=base_channels).to(torch_device)
    checkpoint_is_qat = checkpoint_quantization_mode.startswith("qat")
    prepared_for_qat = False
    if is_qat_mode and checkpoint_is_qat:
        model = prepare_model_for_qat(model=model, backend=effective_qat_backend)
        prepared_for_qat = True

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=selected_learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    start_epoch = 1
    best_val_loss = float("inf")

    if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
        load_info = load_checkpoint_flexible(
            model=model,
            checkpoint_path=resume_checkpoint_path,
            device=torch_device,
            optimizer=None if is_qat_mode else optimizer,
            allow_partial_state_dict=is_qat_mode or checkpoint_is_qat,
        )

        start_epoch = load_info["start_epoch"]
        best_val_loss = load_info["best_val_loss"]

    if is_qat_mode and not prepared_for_qat:
        model = prepare_model_for_qat(model=model, backend=effective_qat_backend)
        prepared_for_qat = True

    if is_qat_mode:
        optimizer = torch.optim.AdamW(model.parameters(), lr=selected_learning_rate)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
            print("QAT 模式下将忽略历史 optimizer 状态，并以当前学习率重新初始化优化器。")

    metadata = build_model_metadata(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        input_width=input_width,
        input_height=input_height,
        preprocess_mode=str(resolved_model_metadata["preprocess_mode"]),
        amp=use_amp,
        use_mask=use_mask,
        quantization_mode="qat_fine_tune" if is_qat_mode else "fp32",
        qat_backend=effective_qat_backend,
    )

    train_start_epoch = start_epoch
    train_end_epoch = num_epochs
    if is_qat_mode:
        train_end_epoch = start_epoch + num_epochs - 1

    for epoch in range(train_start_epoch, train_end_epoch + 1):
        if is_qat_mode:
            qat_stage_epoch = epoch - train_start_epoch + 1
            qat_state = update_qat_epoch_state(
                model=model,
                stage_epoch=qat_stage_epoch,
                total_stage_epochs=num_epochs,
                disable_observer_last_n_epochs=qat_disable_observer_last_n_epochs,
                freeze_bn_after_epoch=qat_freeze_bn_after_epoch,
            )
            print(
                "QAT state | "
                f"stage_epoch={qat_state['stage_epoch']}/{qat_state['total_stage_epochs']} "
                f"observer_enabled={qat_state['observer_enabled']} "
                f"bn_frozen={qat_state['bn_frozen']}"
            )

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=torch_device,
            use_amp=use_amp,
            num_classes=num_classes,
            scaler=scaler,
        )

        val_metrics = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=torch_device,
            use_amp=use_amp,
            num_classes=num_classes,
        )

        print_metrics(epoch, train_metrics, val_metrics)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]

            save_full_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_loss=best_val_loss,
                save_path=save_checkpoint_path,
                metadata=metadata,
            )

            print("已保存当前最佳完整 checkpoint。")


def parse_args() -> argparse.Namespace:
    """
    summary: 解析训练命令行参数
    param 无: 无
    return: 参数对象
    """
    parser = argparse.ArgumentParser(description="训练轻量 U-Net 眼部分割模型")

    parser.add_argument("--root_dir", type=str, required=True, help="OpenEDS 风格数据集根目录")
    parser.add_argument("--batch_size", type=int, default=4, help="训练批大小")
    parser.add_argument("--num_epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="优化器学习率")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader 进程数")

    parser.add_argument("--save_checkpoint_path", type=str, default=DEFAULT_CHECKPOINT_PATH, help="最佳模型 checkpoint 保存路径")
    parser.add_argument("--resume_checkpoint_path", type=str, default=None, help="可选续训 checkpoint 路径")

    parser.add_argument("--in_channels", type=int, default=DEFAULT_IN_CHANNELS, help="模型输入通道数")
    parser.add_argument("--num_classes", type=int, default=DEFAULT_NUM_CLASSES, help="模型输出类别数")
    parser.add_argument("--base_channels", type=int, default=DEFAULT_BASE_CHANNELS, help="U-Net 基础通道数")
    parser.add_argument("--input_width", type=int, default=DEFAULT_INPUT_WIDTH, help="模型输入宽度")
    parser.add_argument("--input_height", type=int, default=DEFAULT_INPUT_HEIGHT, help="模型输入高度")
    parser.add_argument("--device", type=str, default="auto", help="运行设备: auto/cpu/cuda")
    parser.add_argument("--amp", action="store_true", default=DEFAULT_USE_AMP, help="启用 CUDA AMP")
    parser.add_argument("--no-amp", action="store_false", dest="amp", help="禁用 CUDA AMP")
    parser.add_argument("--use_mask", action="store_true", default=DEFAULT_USE_MASK, help="读取 mask 并计算 masked_acc")
    parser.add_argument("--no-use_mask", action="store_false", dest="use_mask", help="不读取 mask，加快数据加载")
    parser.add_argument("--qat_mode", type=str, default="off", choices=["off", "fine_tune"], help="QAT 模式，off 或 fine_tune")
    parser.add_argument(
        "--qat_backend",
        type=str,
        default=DEFAULT_QAT_BACKEND,
        choices=["qnnpack", "fbgemm"],
        help="QAT backend，建议部署到 ARM 时使用 qnnpack",
    )
    parser.add_argument("--qat_learning_rate", type=float, default=None, help="QAT 模式下可选学习率覆盖")
    parser.add_argument(
        "--qat_disable_observer_last_n_epochs",
        type=int,
        default=1,
        help="QAT 最后多少轮关闭 observer",
    )
    parser.add_argument(
        "--qat_freeze_bn_after_epoch",
        type=int,
        default=2,
        help="QAT 从第几轮开始冻结 BN 统计，<=0 表示不冻结",
    )

    return parser.parse_args()


def main() -> None:
    """
    summary: 训练入口
    param 无: 无
    return: 无
    """
    args = parse_args()

    run_training(
        root_dir=args.root_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        save_checkpoint_path=args.save_checkpoint_path,
        resume_checkpoint_path=args.resume_checkpoint_path,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        base_channels=args.base_channels,
        input_width=args.input_width,
        input_height=args.input_height,
        use_amp=args.amp,
        use_mask=args.use_mask,
        device=args.device,
        qat_mode=args.qat_mode,
        qat_backend=args.qat_backend,
        qat_learning_rate=args.qat_learning_rate,
        qat_disable_observer_last_n_epochs=args.qat_disable_observer_last_n_epochs,
        qat_freeze_bn_after_epoch=args.qat_freeze_bn_after_epoch,
    )
