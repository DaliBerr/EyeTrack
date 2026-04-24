import argparse
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from eyetrack.config import (
    DEFAULT_BASE_CHANNELS,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_IN_CHANNELS,
    DEFAULT_INPUT_HEIGHT,
    DEFAULT_INPUT_WIDTH,
    DEFAULT_NUM_CLASSES,
    DEFAULT_ONNX_PATH,
    DEFAULT_USE_AMP,
)
from eyetrack.data.openeds import OpenEDSSegDataset
from eyetrack.deployment.onnx_tools import evaluate_onnx_segmentation
from eyetrack.metrics.segmentation import average_dict_values, compute_dice_per_class
from eyetrack.models.unet import UNet
from eyetrack.runtime import autocast_context, resolve_device
from eyetrack.training.checkpoints import load_checkpoint_flexible, resolve_model_metadata
from eyetrack.training.qat_engine import prepare_model_for_qat, strip_prepared_qat_model_to_float


def compare_onnx_with_pytorch(
    model_path: str,
    checkpoint_path: str,
    root_dir: str,
    split: str = "validation",
    input_width: int = DEFAULT_INPUT_WIDTH,
    input_height: int = DEFAULT_INPUT_HEIGHT,
    in_channels: int = DEFAULT_IN_CHANNELS,
    num_classes: int = DEFAULT_NUM_CLASSES,
    base_channels: int = DEFAULT_BASE_CHANNELS,
    use_amp: bool = DEFAULT_USE_AMP,
    device: str = "auto",
    batch_size: int = 1,
    num_workers: int = 0,
    limit: int | None = None,
) -> Dict[str, float]:
    """
    summary: PyTorch checkpoint ONNX model output
    param model_path: ONNX modelpath
    param checkpoint_path: PyTorch checkpoint path
    param root_dir: dataset directory
    param split:
    param input_width: modelinput
    param input_height: modelinput
    param in_channels: input
    param num_classes: classcount
    param base_channels: U-Net
    param use_amp: enable AMP
    param device:
    param batch_size: DataLoader batch
    param num_workers: DataLoader
    param limit: sample
    return:
    """
    import onnxruntime as ort

    resolved_model_metadata = resolve_model_metadata(
        checkpoint_path=checkpoint_path,
        device="cpu",
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        input_width=input_width,
        input_height=input_height,
        amp=use_amp,
    )

    dataset = OpenEDSSegDataset(
        root_dir=root_dir,
        split=split,
        input_width=int(resolved_model_metadata["input_width"]),
        input_height=int(resolved_model_metadata["input_height"]),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    torch_device = resolve_device(device)
    checkpoint_quantization_mode = str(resolved_model_metadata["quantization_mode"])
    checkpoint_qat_backend = str(resolved_model_metadata["qat_backend"])
    checkpoint_is_qat = checkpoint_quantization_mode.startswith("qat")
    effective_qat_backend = checkpoint_qat_backend if checkpoint_qat_backend in {"qnnpack", "fbgemm"} else "qnnpack"
    model = UNet(
        in_channels=int(resolved_model_metadata["in_channels"]),
        num_classes=int(resolved_model_metadata["num_classes"]),
        base_channels=int(resolved_model_metadata["base_channels"]),
    ).to(torch_device)
    if checkpoint_is_qat:
        model = prepare_model_for_qat(model=model, backend=effective_qat_backend)

    load_info = load_checkpoint_flexible(
        model=model,
        checkpoint_path=checkpoint_path,
        device=torch_device,
        optimizer=None,
        allow_partial_state_dict=checkpoint_is_qat,
    )
    print("PyTorch model metadata:", resolved_model_metadata)
    print("PyTorch checkpoint info:", load_info)
    if checkpoint_is_qat:
        model.apply(torch.ao.quantization.disable_observer)
        freeze_bn_fn = getattr(torch.ao.quantization, "freeze_bn_stats", None)
        if freeze_bn_fn is not None:
            model.apply(freeze_bn_fn)
        model = strip_prepared_qat_model_to_float(model).to(torch_device)
        print("PyTorch model QAT checkpoint model.")
    model.eval()

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    total_agreement = 0.0
    total_batches = 0
    dice_list = []

    with torch.inference_mode():
        for batch in dataloader:
            images = batch["image"].to(torch_device)
            onnx_inputs = batch["image"].cpu().numpy().astype(np.float32)

            with autocast_context(device=torch_device, use_amp=bool(resolved_model_metadata["amp"])):
                torch_logits = model(images)
            torch_preds = torch.argmax(torch_logits, dim=1).cpu()

            onnx_logits = session.run(None, {input_name: onnx_inputs})[0]
            onnx_preds = torch.from_numpy(np.argmax(onnx_logits, axis=1).astype(np.int64))

            agreement = (torch_preds == onnx_preds).float().mean().item()
            dice = compute_dice_per_class(
                onnx_preds,
                torch_preds,
                num_classes=int(resolved_model_metadata["num_classes"]),
                ignore_background=True,
            )

            total_agreement += agreement
            total_batches += 1
            dice_list.append(dice)

            if limit is not None and batch_size * total_batches >= limit:
                break

    avg_dice = average_dict_values(dice_list)
    return {
        "prediction_agreement": total_agreement / max(total_batches, 1),
        "compare_dice_1": avg_dice.get(1, 0.0),
        "compare_dice_2": avg_dice.get(2, 0.0),
        "compare_dice_3": avg_dice.get(3, 0.0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=" ONNX model, optional PyTorch checkpoint ")
    parser.add_argument("--model_path", type=str, default=DEFAULT_ONNX_PATH, help="ONNX modelpath")
    parser.add_argument("--root_dir", type=str, required=True, help="OpenEDS dataset directory")
    parser.add_argument("--split", type=str, default="validation", help=" ")
    parser.add_argument("--backend", type=str, default="cpu", choices=["cpu", "nnapi"], help="ONNX Runtime backend")
    parser.add_argument("--input_width", type=int, default=DEFAULT_INPUT_WIDTH, help="modelinput ")
    parser.add_argument("--input_height", type=int, default=DEFAULT_INPUT_HEIGHT, help="modelinput ")
    parser.add_argument("--batch_size", type=int, default=1, help=" batch ")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader ")
    parser.add_argument("--limit", type=int, default=None, help=" sample")

    parser.add_argument("--checkpoint_path", type=str, default=None, help="optional PyTorch checkpoint, ")
    parser.add_argument("--in_channels", type=int, default=DEFAULT_IN_CHANNELS, help="modelinput ")
    parser.add_argument("--num_classes", type=int, default=DEFAULT_NUM_CLASSES, help="modeloutputclass ")
    parser.add_argument("--base_channels", type=int, default=DEFAULT_BASE_CHANNELS, help="U-Net ")
    parser.add_argument("--device", type=str, default="auto", help="PyTorch ")
    parser.add_argument("--amp", action="store_true", default=DEFAULT_USE_AMP, help="PyTorch enable AMP")
    parser.add_argument("--no-amp", action="store_false", dest="amp", help="PyTorch disable AMP")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    onnx_metrics = evaluate_onnx_segmentation(
        model_path=args.model_path,
        root_dir=args.root_dir,
        split=args.split,
        input_width=args.input_width,
        input_height=args.input_height,
        num_classes=args.num_classes,
        backend=args.backend,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        limit=args.limit,
    )
    print("ONNX metrics:", onnx_metrics)

    if args.checkpoint_path is None:
        return

    compare_metrics = compare_onnx_with_pytorch(
        model_path=args.model_path,
        checkpoint_path=args.checkpoint_path,
        root_dir=args.root_dir,
        split=args.split,
        input_width=args.input_width,
        input_height=args.input_height,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        base_channels=args.base_channels,
        use_amp=args.amp,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        limit=args.limit,
    )
    print("PyTorch vs ONNX compare metrics:", compare_metrics)


if __name__ == "__main__":
    main()
