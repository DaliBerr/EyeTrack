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
    summary:
    param seed:
    return: none
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_metrics(epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
    """
    summary: training validation
    param epoch: current
    param train_metrics: training
    param val_metrics: validation
    return: none
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
    summary: training OpenEDS model supports AMP
    param root_dir: dataset directory
    param batch_size: batch
    param num_epochs: training
    param learning_rate:
    param num_workers: DataLoader
    param save_checkpoint_path: checkpoint savepath
    param resume_checkpoint_path: optional checkpoint
    param in_channels: input
    param num_classes: classcount
    param base_channels: U-Net
    param input_width: modelinput
    param input_height: modelinput
    param use_amp: enable AMP
    param use_mask: read mask masked_acc
    param device:
    param qat_mode: QAT, off fine_tune
    param qat_backend: QAT backend, qnnpack fbgemm
    param qat_learning_rate: QAT optional
    param qat_disable_observer_last_n_epochs: QAT observer
    param qat_freeze_bn_after_epoch: QAT BN
    return: none
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
        print("QAT disable AMP quantization.")
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
            print("QAT optimizer, current.")

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

            print(" savecurrent checkpoint.")


def parse_args() -> argparse.Namespace:
    """
    summary: parsetrainingCLIarguments
    param none: none
    return: arguments
    """
    parser = argparse.ArgumentParser(description="training U-Net model")

    parser.add_argument("--root_dir", type=str, required=True, help="OpenEDS dataset directory")
    parser.add_argument("--batch_size", type=int, default=4, help="trainingbatch ")
    parser.add_argument("--num_epochs", type=int, default=10, help="training ")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help=" ")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader ")

    parser.add_argument("--save_checkpoint_path", type=str, default=DEFAULT_CHECKPOINT_PATH, help=" model checkpoint savepath")
    parser.add_argument("--resume_checkpoint_path", type=str, default=None, help="optional checkpoint path")

    parser.add_argument("--in_channels", type=int, default=DEFAULT_IN_CHANNELS, help="modelinput ")
    parser.add_argument("--num_classes", type=int, default=DEFAULT_NUM_CLASSES, help="modeloutputclass ")
    parser.add_argument("--base_channels", type=int, default=DEFAULT_BASE_CHANNELS, help="U-Net ")
    parser.add_argument("--input_width", type=int, default=DEFAULT_INPUT_WIDTH, help="modelinput ")
    parser.add_argument("--input_height", type=int, default=DEFAULT_INPUT_HEIGHT, help="modelinput ")
    parser.add_argument("--device", type=str, default="auto", help=": auto/cpu/cuda")
    parser.add_argument("--amp", action="store_true", default=DEFAULT_USE_AMP, help="enable CUDA AMP")
    parser.add_argument("--no-amp", action="store_false", dest="amp", help="disable CUDA AMP")
    parser.add_argument("--use_mask", action="store_true", default=DEFAULT_USE_MASK, help="read mask masked_acc")
    parser.add_argument("--no-use_mask", action="store_false", dest="use_mask", help=" read mask, ")
    parser.add_argument("--qat_mode", type=str, default="off", choices=["off", "fine_tune"], help="QAT, off fine_tune")
    parser.add_argument(
        "--qat_backend",
        type=str,
        default=DEFAULT_QAT_BACKEND,
        choices=["qnnpack", "fbgemm"],
        help="QAT backend, ARM qnnpack",
    )
    parser.add_argument("--qat_learning_rate", type=float, default=None, help="QAT optional ")
    parser.add_argument(
        "--qat_disable_observer_last_n_epochs",
        type=int,
        default=1,
        help="QAT observer",
    )
    parser.add_argument(
        "--qat_freeze_bn_after_epoch",
        type=int,
        default=2,
        help="QAT BN, <=0 ",
    )

    return parser.parse_args()


def main() -> None:
    """
    summary: training
    param none: none
    return: none
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
