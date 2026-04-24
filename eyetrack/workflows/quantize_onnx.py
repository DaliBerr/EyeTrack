import argparse
from pathlib import Path
from typing import Any

from eyetrack.config import DEFAULT_INPUT_HEIGHT, DEFAULT_INPUT_WIDTH, DEFAULT_INT8_ONNX_PATH, DEFAULT_ONNX_PATH
from eyetrack.deployment.onnx_tools import (
    CalibrationImageReader,
    compute_mean_foreground_dice,
    evaluate_onnx_segmentation,
    get_onnx_input_name,
    preprocess_onnx_model_file,
    require_onnxruntime,
    require_onnxruntime_quantization,
    resolve_image_dir,
)


def parse_quant_enum(enum_cls, value: str):
    name = value.upper()
    for candidate in dir(enum_cls):
        if candidate.upper() == name:
            return getattr(enum_cls, candidate)
    valid = [candidate for candidate in dir(enum_cls) if not candidate.startswith("_")]
    raise ValueError(f"unsupported quantization: {value}, optional: {valid}")


def is_histogram_calibration_method(calibration_method: str) -> bool:
    return calibration_method.strip().lower() in {"percentile", "entropy", "distribution"}


def is_probable_calibration_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "bad allocation" in message or
        "out of memory" in message or
        "failed to allocate" in message or
        "unable to allocate" in message or
        "arraymemoryerror" in message
    )


def resolve_effective_calibration_stride(reader: CalibrationImageReader, calibration_stride: int | None) -> int | None:
    if calibration_stride is None or calibration_stride <= 0:
        return None

    total = len(reader)
    if total == 0:
        return None

    stride = min(calibration_stride, total)
    while stride > 1 and total % stride != 0:
        stride -= 1
    return max(stride, 1)


def build_quantize_static_extra_options(
    reader: CalibrationImageReader,
    calibration_method: str,
    calibration_stride: int | None,
) -> dict[str, Any]:
    """
    summary: ORT quantization extra_options
    param reader: calibration reader
    param calibration_method: calibration
    param calibration_stride: calibration stride
    return: extra_options dict
    """
    effective_stride = resolve_effective_calibration_stride(reader=reader, calibration_stride=calibration_stride)
    extra_options: dict[str, Any] = {}
    if effective_stride is not None:
        # ORT 1.24.x calibration stride reader,.
        extra_options["CalibStridedMinMax"] = effective_stride
    return extra_options


def build_calibration_reader(
    calibration_root: str,
    split: str,
    input_name: str,
    input_width: int,
    input_height: int,
    calibration_limit: int,
) -> CalibrationImageReader:
    calibration_image_dir = resolve_image_dir(root_dir=calibration_root, split=split)
    return CalibrationImageReader(
        image_dir=calibration_image_dir,
        input_name=input_name,
        input_width=input_width,
        input_height=input_height,
        limit=calibration_limit,
    )


def run_quantize_static(
    quantization: Any,
    model_path: str,
    output_path: str,
    reader: CalibrationImageReader,
    quant_format: str,
    activation_type: str,
    weight_type: str,
    calibration_method: str,
    per_channel: bool,
    extra_options: dict[str, Any] | None = None,
) -> None:
    quantization.quantize_static(
        model_input=model_path,
        model_output=output_path,
        calibration_data_reader=reader,
        quant_format=parse_quant_enum(quantization.QuantFormat, quant_format),
        activation_type=parse_quant_enum(quantization.QuantType, activation_type),
        weight_type=parse_quant_enum(quantization.QuantType, weight_type),
        per_channel=per_channel,
        calibrate_method=parse_quant_enum(quantization.CalibrationMethod, calibration_method),
        extra_options=extra_options,
    )


def quantize_onnx_model(
    model_path: str = DEFAULT_ONNX_PATH,
    calibration_root: str | None = None,
    output_path: str = DEFAULT_INT8_ONNX_PATH,
    split: str = "train",
    input_width: int = DEFAULT_INPUT_WIDTH,
    input_height: int = DEFAULT_INPUT_HEIGHT,
    quant_format: str = "qdq",
    activation_type: str = "qint8",
    weight_type: str = "qint8",
    calibration_method: str = "percentile",
    calibration_limit: int = 256,
    calibration_stride: int = 1,
    per_channel: bool = True,
    auto_fallback_to_minmax_on_oom: bool = True,
    oom_fallback_calibration_limit: int = 32,
    auto_fallback_to_u8u8: bool = False,
    fallback_output_path: str | None = None,
    validation_root: str | None = None,
    validation_split: str = "validation",
    max_dice_loss: float = 0.03,
) -> None:
    """
    summary: ONNX model PTQ, optional fallback U8U8
    param model_path: ONNX modelpath
    param calibration_root: OpenEDS calibration directory
    param output_path: INT8 ONNX outputpath
    param split: calibration
    param input_width: modelinput
    param input_height: modelinput
    param quant_format: quantization
    param activation_type: quantization
    param weight_type: quantization
    param calibration_method: calibration
    param calibration_limit: calibrationsample
    param calibration_stride: calibration stride, 1 samplecalibration
    param per_channel: enable per-channel
    param auto_fallback_to_minmax_on_oom: histogram calibration OOM, fallback MinMax
    param oom_fallback_calibration_limit: OOM fallback calibrationsample
    param auto_fallback_to_u8u8: fallback U8U8
    param fallback_output_path: fallbackmodeloutputpath
    param validation_root: fallback validation directory
    param validation_split: validation
    param max_dice_loss: maximum Dice
    return: none
    """
    if calibration_root is None:
        raise ValueError("--calibration_root is required for static PTQ calibration.")

    require_onnxruntime()
    quantization = require_onnxruntime_quantization()

    preprocess_onnx_model_file(model_path)

    input_name = get_onnx_input_name(model_path)
    reader = build_calibration_reader(
        calibration_root=calibration_root,
        split=split,
        input_name=input_name,
        input_width=input_width,
        input_height=input_height,
        calibration_limit=calibration_limit,
    )
    extra_options = build_quantize_static_extra_options(
        reader=reader,
        calibration_method=calibration_method,
        calibration_stride=calibration_stride,
    )
    if extra_options.get("CalibStridedMinMax") is not None:
        print(
            "calibration: "
            f"method={calibration_method} stride={extra_options['CalibStridedMinMax']} samples={len(reader)}"
        )

    output_model_path = Path(output_path)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    effective_calibration_method = calibration_method
    effective_extra_options = extra_options

    try:
        run_quantize_static(
            quantization=quantization,
            model_path=model_path,
            output_path=str(output_model_path),
            reader=reader,
            quant_format=quant_format,
            activation_type=activation_type,
            weight_type=weight_type,
            calibration_method=calibration_method,
            per_channel=per_channel,
            extra_options=extra_options,
        )
    except Exception as exc:
        if (
            auto_fallback_to_minmax_on_oom and
            is_histogram_calibration_method(calibration_method) and
            is_probable_calibration_oom(exc)
        ):
            fallback_limit = min(calibration_limit, oom_fallback_calibration_limit)
            print(
                " histogram calibration, "
                f" fallback MinMax, calibrationsample {fallback_limit}."
            )
            reader = build_calibration_reader(
                calibration_root=calibration_root,
                split=split,
                input_name=input_name,
                input_width=input_width,
                input_height=input_height,
                calibration_limit=fallback_limit,
            )
            fallback_extra_options = build_quantize_static_extra_options(
                reader=reader,
                calibration_method="minmax",
                calibration_stride=calibration_stride,
            )
            run_quantize_static(
                quantization=quantization,
                model_path=model_path,
                output_path=str(output_model_path),
                reader=reader,
                quant_format=quant_format,
                activation_type=activation_type,
                weight_type=weight_type,
                calibration_method="minmax",
                per_channel=per_channel,
                extra_options=fallback_extra_options,
            )
            effective_calibration_method = "minmax"
            effective_extra_options = fallback_extra_options
            print("Completed MinMax OOM fallback quantization.")
        else:
            if is_histogram_calibration_method(calibration_method) and is_probable_calibration_oom(exc):
                raise RuntimeError(
                    "currentcalibration ONNX Runtime histogram calibration."
                    " --calibration_stride 1; "
                    ", --calibration_limit 16 32; "
                    " --calibration_method minmax; "
                    " --auto_fallback_to_minmax_on_oom."
                ) from exc
            raise
    print(f" quantizationmodel: {output_model_path}")

    if not auto_fallback_to_u8u8 or validation_root is None:
        return

    baseline_metrics = evaluate_onnx_segmentation(
        model_path=model_path,
        root_dir=validation_root,
        split=validation_split,
        input_width=input_width,
        input_height=input_height,
        backend="cpu",
    )
    quant_metrics = evaluate_onnx_segmentation(
        model_path=str(output_model_path),
        root_dir=validation_root,
        split=validation_split,
        input_width=input_width,
        input_height=input_height,
        backend="cpu",
    )

    baseline_dice = compute_mean_foreground_dice(baseline_metrics)
    quant_dice = compute_mean_foreground_dice(quant_metrics)
    dice_loss = baseline_dice - quant_dice

    print("FP32 ONNX metrics:", baseline_metrics)
    print("INT8 QDQ metrics:", quant_metrics)
    print(f"mean foreground dice loss: {dice_loss:.6f}")

    if dice_loss <= max_dice_loss:
        print("quantization threshold, none U8U8 fallback.")
        return

    fallback_path = fallback_output_path
    if fallback_path is None:
        fallback_path = str(output_model_path.with_name(output_model_path.stem + "_u8u8.onnx"))

    reader.rewind()
    quantization.quantize_static(
        model_input=model_path,
        model_output=fallback_path,
        calibration_data_reader=reader,
        quant_format=quantization.QuantFormat.QDQ,
        activation_type=quantization.QuantType.QUInt8,
        weight_type=quantization.QuantType.QUInt8,
        per_channel=per_channel,
        calibrate_method=parse_quant_enum(quantization.CalibrationMethod, effective_calibration_method),
        extra_options=effective_extra_options,
    )
    print(f"QInt8 threshold, U8U8 fallback model: {fallback_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=" ONNX model PTQ quantization")
    parser.add_argument("--model_path", type=str, default=DEFAULT_ONNX_PATH, help=" ONNX modelpath")
    parser.add_argument("--calibration_root", type=str, required=True, help="OpenEDS calibration directory")
    parser.add_argument("--output_path", type=str, default=DEFAULT_INT8_ONNX_PATH, help="INT8 ONNX outputpath")
    parser.add_argument("--split", type=str, default="train", help="calibration ")
    parser.add_argument("--input_width", type=int, default=DEFAULT_INPUT_WIDTH, help="modelinput ")
    parser.add_argument("--input_height", type=int, default=DEFAULT_INPUT_HEIGHT, help="modelinput ")
    parser.add_argument("--quant_format", type=str, default="qdq", help="quantization, default qdq")
    parser.add_argument("--activation_type", type=str, default="qint8", help=" quantization ")
    parser.add_argument("--weight_type", type=str, default="qint8", help=" quantization ")
    parser.add_argument("--calibration_method", type=str, default="percentile", help="calibration ")
    parser.add_argument("--calibration_limit", type=int, default=256, help=" calibration maximumimage ")
    parser.add_argument(
        "--calibration_stride",
        type=int,
        default=1,
        help=" calibration stride; 1 samplecalibration, ; <=0 ",
    )
    parser.add_argument("--per_channel", action="store_true", default=True, help="enable per-channel quantization")
    parser.add_argument("--no-per_channel", action="store_false", dest="per_channel", help="disable per-channel quantization")
    parser.add_argument(
        "--auto_fallback_to_minmax_on_oom",
        action="store_true",
        dest="auto_fallback_to_minmax_on_oom",
        help="Histogram calibration OOM fallback MinMax",
    )
    parser.add_argument(
        "--no-auto_fallback_to_minmax_on_oom",
        action="store_false",
        dest="auto_fallback_to_minmax_on_oom",
        help="disable Histogram calibration OOM MinMax fallback",
    )
    parser.set_defaults(auto_fallback_to_minmax_on_oom=True)
    parser.add_argument("--oom_fallback_calibration_limit", type=int, default=32, help="OOM fallback MinMax calibrationsample ")
    parser.add_argument("--auto_fallback_to_u8u8", action="store_true", help=" fallback U8U8")
    parser.add_argument("--fallback_output_path", type=str, default=None, help="U8U8 fallback outputpath")
    parser.add_argument("--validation_root", type=str, default=None, help=" fallback validation directory")
    parser.add_argument("--validation_split", type=str, default="validation", help="validation ")
    parser.add_argument("--max_dice_loss", type=float, default=0.03, help="maximum Dice ")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quantize_onnx_model(
        model_path=args.model_path,
        calibration_root=args.calibration_root,
        output_path=args.output_path,
        split=args.split,
        input_width=args.input_width,
        input_height=args.input_height,
        quant_format=args.quant_format,
        activation_type=args.activation_type,
        weight_type=args.weight_type,
        calibration_method=args.calibration_method,
        calibration_limit=args.calibration_limit,
        calibration_stride=args.calibration_stride,
        per_channel=args.per_channel,
        auto_fallback_to_minmax_on_oom=args.auto_fallback_to_minmax_on_oom,
        oom_fallback_calibration_limit=args.oom_fallback_calibration_limit,
        auto_fallback_to_u8u8=args.auto_fallback_to_u8u8,
        fallback_output_path=args.fallback_output_path,
        validation_root=args.validation_root,
        validation_split=args.validation_split,
        max_dice_loss=args.max_dice_loss,
    )


if __name__ == "__main__":
    main()
