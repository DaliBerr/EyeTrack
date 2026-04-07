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
    raise ValueError(f"不支持的量化选项: {value}，可选值: {valid}")


def is_histogram_calibration_method(calibration_method: str) -> bool:
    return calibration_method.strip().lower() in {"percentile", "entropy", "distribution"}


def is_probable_calibration_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "bad allocation" in message or "out of memory" in message or "failed to allocate" in message


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
    summary: 对 ONNX 模型执行静态 PTQ，并可选自动回退到 U8U8
    param model_path: 浮点 ONNX 模型路径
    param calibration_root: OpenEDS 风格校准数据根目录
    param output_path: INT8 ONNX 输出路径
    param split: 校准数据划分
    param input_width: 模型输入宽度
    param input_height: 模型输入高度
    param quant_format: 量化格式
    param activation_type: 激活量化类型
    param weight_type: 权重量化类型
    param calibration_method: 校准方法
    param calibration_limit: 校准样本上限
    param per_channel: 是否启用 per-channel
    param auto_fallback_to_minmax_on_oom: 若 histogram 校准 OOM，是否自动回退到 MinMax
    param oom_fallback_calibration_limit: OOM 回退时使用的校准样本上限
    param auto_fallback_to_u8u8: 是否自动回退到 U8U8
    param fallback_output_path: 回退模型输出路径
    param validation_root: 用于自动回退判断的验证集根目录
    param validation_split: 验证数据划分
    param max_dice_loss: 最大可接受 Dice 损失
    return: 无
    """
    if calibration_root is None:
        raise ValueError("必须提供 --calibration_root，用于静态 PTQ 校准。")

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

    output_model_path = Path(output_path)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    effective_calibration_method = calibration_method

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
        )
    except Exception as exc:
        if (
            auto_fallback_to_minmax_on_oom and
            is_histogram_calibration_method(calibration_method) and
            is_probable_calibration_oom(exc)
        ):
            fallback_limit = min(calibration_limit, oom_fallback_calibration_limit)
            print(
                "检测到 histogram 校准阶段内存不足，"
                f"将自动回退到 MinMax，并把校准样本数限制为 {fallback_limit}。"
            )
            reader = build_calibration_reader(
                calibration_root=calibration_root,
                split=split,
                input_name=input_name,
                input_width=input_width,
                input_height=input_height,
                calibration_limit=fallback_limit,
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
            )
            effective_calibration_method = "minmax"
            print("已完成 MinMax OOM fallback 量化。")
        else:
            if is_histogram_calibration_method(calibration_method) and is_probable_calibration_oom(exc):
                raise RuntimeError(
                    "当前校准配置触发了 ONNX Runtime histogram 校准器的内存问题。"
                    "建议先把 --calibration_limit 降到 16 或 32；"
                    "若仍失败，改用 --calibration_method minmax；"
                    "或者直接加 --auto_fallback_to_minmax_on_oom。"
                ) from exc
            raise
    print(f"已生成量化模型: {output_model_path}")

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
        print("量化结果满足精度阈值，无需执行 U8U8 fallback。")
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
    )
    print(f"QInt8 结果超出阈值，已生成 U8U8 fallback 模型: {fallback_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对 ONNX 模型执行静态 PTQ 量化")
    parser.add_argument("--model_path", type=str, default=DEFAULT_ONNX_PATH, help="浮点 ONNX 模型路径")
    parser.add_argument("--calibration_root", type=str, required=True, help="OpenEDS 风格校准数据根目录")
    parser.add_argument("--output_path", type=str, default=DEFAULT_INT8_ONNX_PATH, help="INT8 ONNX 输出路径")
    parser.add_argument("--split", type=str, default="train", help="校准数据划分")
    parser.add_argument("--input_width", type=int, default=DEFAULT_INPUT_WIDTH, help="模型输入宽度")
    parser.add_argument("--input_height", type=int, default=DEFAULT_INPUT_HEIGHT, help="模型输入高度")
    parser.add_argument("--quant_format", type=str, default="qdq", help="量化格式，默认 qdq")
    parser.add_argument("--activation_type", type=str, default="qint8", help="激活量化类型")
    parser.add_argument("--weight_type", type=str, default="qint8", help="权重量化类型")
    parser.add_argument("--calibration_method", type=str, default="percentile", help="校准方法")
    parser.add_argument("--calibration_limit", type=int, default=256, help="用于校准的最大图像数")
    parser.add_argument("--per_channel", action="store_true", default=True, help="启用 per-channel 权重量化")
    parser.add_argument("--no-per_channel", action="store_false", dest="per_channel", help="禁用 per-channel 权重量化")
    parser.add_argument(
        "--auto_fallback_to_minmax_on_oom",
        action="store_true",
        dest="auto_fallback_to_minmax_on_oom",
        help="Histogram 校准 OOM 时自动回退到 MinMax",
    )
    parser.add_argument(
        "--no-auto_fallback_to_minmax_on_oom",
        action="store_false",
        dest="auto_fallback_to_minmax_on_oom",
        help="禁用 Histogram 校准 OOM 时的自动 MinMax 回退",
    )
    parser.set_defaults(auto_fallback_to_minmax_on_oom=True)
    parser.add_argument("--oom_fallback_calibration_limit", type=int, default=32, help="OOM 回退到 MinMax 时使用的校准样本上限")
    parser.add_argument("--auto_fallback_to_u8u8", action="store_true", help="若精度下降过大则自动回退到 U8U8")
    parser.add_argument("--fallback_output_path", type=str, default=None, help="U8U8 fallback 输出路径")
    parser.add_argument("--validation_root", type=str, default=None, help="用于自动回退判断的验证集根目录")
    parser.add_argument("--validation_split", type=str, default="validation", help="验证数据划分")
    parser.add_argument("--max_dice_loss", type=float, default=0.03, help="最大可接受平均前景 Dice 损失")
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
