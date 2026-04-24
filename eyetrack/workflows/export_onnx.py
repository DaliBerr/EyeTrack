import argparse
from pathlib import Path

import torch

from eyetrack.config import (
    DEFAULT_BASE_CHANNELS,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_IN_CHANNELS,
    DEFAULT_INPUT_HEIGHT,
    DEFAULT_INPUT_WIDTH,
    DEFAULT_NUM_CLASSES,
    DEFAULT_ONNX_PATH,
    DEFAULT_QAT_BACKEND,
    DEFAULT_QUANTIZATION_MODE,
    build_model_metadata,
)
from eyetrack.deployment.onnx_tools import preprocess_onnx_model_file, require_onnx
from eyetrack.models.unet import UNet
from eyetrack.runtime import resolve_device
from eyetrack.training.checkpoints import load_checkpoint_flexible, resolve_model_metadata
from eyetrack.training.qat_engine import prepare_model_for_qat, strip_prepared_qat_model_to_float


def export_checkpoint_to_onnx(
    checkpoint_path: str,
    onnx_path: str = DEFAULT_ONNX_PATH,
    in_channels: int = DEFAULT_IN_CHANNELS,
    num_classes: int = DEFAULT_NUM_CLASSES,
    base_channels: int = DEFAULT_BASE_CHANNELS,
    input_width: int = DEFAULT_INPUT_WIDTH,
    input_height: int = DEFAULT_INPUT_HEIGHT,
    device: str = "cpu",
    opset_version: int = 17,
    quantization_mode: str = "fp32_only",
    qat_backend: str = DEFAULT_QAT_BACKEND,
) -> None:
    """
    summary: PyTorch checkpoint input ONNX model
    param checkpoint_path: PyTorch checkpoint path
    param onnx_path: ONNX path
    param in_channels: input
    param num_classes: outputclass
    param base_channels: U-Net
    param input_width: input
    param input_height: input
    param device:
    param opset_version: ONNX opset
    param quantization_mode:, fp32_only qat_qdq_export
    param qat_backend: QAT backend
    return: none
    """
    torch_device = resolve_device(device)
    resolved_model_metadata = resolve_model_metadata(
        checkpoint_path=checkpoint_path,
        device="cpu",
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        input_width=input_width,
        input_height=input_height,
        quantization_mode=DEFAULT_QUANTIZATION_MODE,
        qat_backend=qat_backend,
    )
    print("model metadata:", resolved_model_metadata)

    checkpoint_quantization_mode = str(resolved_model_metadata["quantization_mode"])
    checkpoint_qat_backend = str(resolved_model_metadata["qat_backend"])
    checkpoint_is_qat = checkpoint_quantization_mode.startswith("qat")
    effective_qat_backend = qat_backend
    if checkpoint_qat_backend in {"qnnpack", "fbgemm"}:
        effective_qat_backend = checkpoint_qat_backend

    if quantization_mode == "qat_qdq_export":
        raise RuntimeError(
            "current eager QAT quantization unable to torch.onnx ONNX."
            " --quantization_mode fp32_only QAT checkpoint ONNX, "
            " quantize_onnx_model.py ONNX Runtime quantization QDQ INT8 ONNX."
        )

    model = UNet(
        in_channels=int(resolved_model_metadata["in_channels"]),
        num_classes=int(resolved_model_metadata["num_classes"]),
        base_channels=int(resolved_model_metadata["base_channels"]),
    ).to(torch_device)

    prepared_for_qat = False
    if checkpoint_is_qat:
        model = prepare_model_for_qat(model=model, backend=effective_qat_backend)
        prepared_for_qat = True

    load_info = load_checkpoint_flexible(
        model=model,
        checkpoint_path=checkpoint_path,
        device=torch_device,
        optimizer=None,
        allow_partial_state_dict=checkpoint_is_qat,
    )
    print("checkpoint info:", load_info)

    if checkpoint_is_qat and prepared_for_qat:
        model.apply(torch.ao.quantization.disable_observer)
        freeze_bn_fn = getattr(torch.ao.quantization, "freeze_bn_stats", None)
        if freeze_bn_fn is not None:
            model.apply(freeze_bn_fn)
        model = strip_prepared_qat_model_to_float(model)
        model = model.to(torch_device)
        print(" QAT checkpoint, model.")

    model.eval()

    dummy_input = torch.randn(
        1,
        int(resolved_model_metadata["in_channels"]),
        int(resolved_model_metadata["input_height"]),
        int(resolved_model_metadata["input_width"]),
        device=torch_device,
    )

    output_path = Path(onnx_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=quantization_mode != "qat_qdq_export",
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=None,
    )

    processed_path = preprocess_onnx_model_file(str(output_path))
    onnx = require_onnx()
    model_proto = onnx.load(processed_path)
    metadata = build_model_metadata(
        in_channels=int(resolved_model_metadata["in_channels"]),
        num_classes=int(resolved_model_metadata["num_classes"]),
        base_channels=int(resolved_model_metadata["base_channels"]),
        input_width=int(resolved_model_metadata["input_width"]),
        input_height=int(resolved_model_metadata["input_height"]),
        preprocess_mode=str(resolved_model_metadata["preprocess_mode"]),
        amp=bool(resolved_model_metadata["amp"]),
        use_mask=bool(resolved_model_metadata["use_mask"]),
        quantization_mode="qat_qdq_export" if quantization_mode == "qat_qdq_export" else str(resolved_model_metadata["quantization_mode"]),
        qat_backend=effective_qat_backend,
    )
    onnx.helper.set_model_props(model_proto, {key: str(value) for key, value in metadata.items()})
    onnx.save(model_proto, processed_path)

    print(f"Exported ONNX model: {processed_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=" U-Net checkpoint ONNX model")
    parser.add_argument("--checkpoint_path", type=str, default=DEFAULT_CHECKPOINT_PATH, help="PyTorch checkpoint path")
    parser.add_argument("--onnx_path", type=str, default=DEFAULT_ONNX_PATH, help="ONNX outputpath")
    parser.add_argument("--in_channels", type=int, default=DEFAULT_IN_CHANNELS, help="modelinput ")
    parser.add_argument("--num_classes", type=int, default=DEFAULT_NUM_CLASSES, help="modeloutputclass ")
    parser.add_argument("--base_channels", type=int, default=DEFAULT_BASE_CHANNELS, help="U-Net ")
    parser.add_argument("--input_width", type=int, default=DEFAULT_INPUT_WIDTH, help=" input ")
    parser.add_argument("--input_height", type=int, default=DEFAULT_INPUT_HEIGHT, help=" input ")
    parser.add_argument("--device", type=str, default="cpu", help=", default cpu")
    parser.add_argument("--opset_version", type=int, default=17, help="ONNX opset ")
    parser.add_argument(
        "--quantization_mode",
        type=str,
        default="fp32_only",
        choices=["fp32_only", "qat_qdq_export"],
        help=": default fp32_only; qat_qdq_export current fp32_only + ORT quantization",
    )
    parser.add_argument(
        "--qat_backend",
        type=str,
        default=DEFAULT_QAT_BACKEND,
        choices=["qnnpack", "fbgemm"],
        help="QAT quantized backend",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_checkpoint_to_onnx(
        checkpoint_path=args.checkpoint_path,
        onnx_path=args.onnx_path,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        base_channels=args.base_channels,
        input_width=args.input_width,
        input_height=args.input_height,
        device=args.device,
        opset_version=args.opset_version,
        quantization_mode=args.quantization_mode,
        qat_backend=args.qat_backend,
    )


if __name__ == "__main__":
    main()
