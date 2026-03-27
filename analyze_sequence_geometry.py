import argparse
import csv
import fnmatch
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


METRIC_NAMES = ["norm_dx", "norm_dy", "norm_radius"]


def natural_key(text: str) -> List[Any]:
    """
    summary: 生成自然排序键
    param text: 输入字符串
    return: 可用于排序的键列表
    """
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def parse_optional_float(text: str) -> float:
    """
    summary: 将文本解析为浮点数，空值返回 NaN
    param text: 输入文本
    return: 浮点值或 NaN
    """
    if text is None:
        return float("nan")

    value = text.strip()
    if value == "":
        return float("nan")

    return float(value)


def parse_frame_axis(frame_ids: List[str]) -> np.ndarray:
    """
    summary: 将 frame_id 列表转换为横轴数值
    param frame_ids: 帧编号列表
    return: 数值横轴数组
    """
    axis_values = []

    for index, frame_id in enumerate(frame_ids):
        try:
            axis_values.append(float(frame_id))
        except ValueError:
            axis_values.append(float(index))

    return np.array(axis_values, dtype=np.float64)


def list_sequence_csvs(input_root: str, sequence_glob: str, csv_name: str) -> List[Tuple[str, Path]]:
    """
    summary: 列出所有待分析的序列 csv
    param input_root: 输入根目录
    param sequence_glob: 序列匹配模式
    param csv_name: 目录模式下的 csv 文件名
    return: (序列名, csv 路径) 列表
    """
    root_path = Path(input_root)

    if not root_path.exists():
        raise FileNotFoundError(f"找不到输入根目录: {root_path}")

    items: List[Tuple[str, Path]] = []

    for path in root_path.iterdir():
        if path.is_dir() and fnmatch.fnmatch(path.name, sequence_glob):
            candidate = path / csv_name
            if candidate.exists():
                items.append((path.name, candidate))
        elif path.is_file() and path.suffix.lower() == ".csv":
            stem = path.stem
            if fnmatch.fnmatch(stem, sequence_glob):
                items.append((stem, path))

    items.sort(key=lambda item: natural_key(item[0]))
    return items


def read_sequence_csv(csv_path: Path) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """
    summary: 读取单个序列 csv 中的关键列
    param csv_path: csv 文件路径
    return: frame_id 列表和指标字典
    """
    frame_ids: List[str] = []
    metric_values = {name: [] for name in METRIC_NAMES}

    with open(csv_path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            frame_ids.append(row["frame_id"])
            for metric_name in METRIC_NAMES:
                metric_values[metric_name].append(parse_optional_float(row.get(metric_name, "")))

    metric_arrays = {
        key: np.array(values, dtype=np.float64)
        for key, values in metric_values.items()
    }
    return frame_ids, metric_arrays


def rolling_nanmedian(values: np.ndarray, window_size: int) -> np.ndarray:
    """
    summary: 计算中心对齐的滑动中位数
    param values: 输入数组
    param window_size: 窗口大小
    return: 平滑基线数组
    """
    if window_size <= 1:
        return values.copy()

    half_window = window_size // 2
    output = np.full_like(values, np.nan, dtype=np.float64)

    for index in range(len(values)):
        start = max(0, index - half_window)
        end = min(len(values), index + half_window + 1)
        window = values[start:end]
        finite = np.isfinite(window)
        if finite.any():
            output[index] = np.nanmedian(window)

    return output


def interpolate_nans(values: np.ndarray) -> np.ndarray:
    """
    summary: 对 NaN 做线性插值
    param values: 输入数组
    return: 插值后的数组
    """
    output = values.astype(np.float64).copy()
    valid = np.isfinite(output)

    if not valid.any():
        return output

    valid_index = np.flatnonzero(valid)
    full_index = np.arange(len(output))
    output[~valid] = np.interp(full_index[~valid], valid_index, output[valid])
    return output


def moving_average(values: np.ndarray, window_size: int) -> np.ndarray:
    """
    summary: 计算简单移动平均
    param values: 输入数组
    param window_size: 窗口大小
    return: 平滑后的数组
    """
    if window_size <= 1:
        return values.copy()

    half_window = window_size // 2
    output = np.full_like(values, np.nan, dtype=np.float64)

    for index in range(len(values)):
        start = max(0, index - half_window)
        end = min(len(values), index + half_window + 1)
        window = values[start:end]
        finite = np.isfinite(window)
        if finite.any():
            output[index] = np.nanmean(window)

    return output


def detect_metric_anomalies(
    values: np.ndarray,
    baseline_window: int,
    mad_scale: float,
    min_threshold: float,
) -> np.ndarray:
    """
    summary: 基于滑动中位数残差检测异常点
    param values: 输入指标序列
    param baseline_window: 基线窗口大小
    param mad_scale: MAD 阈值倍数
    param min_threshold: 最小阈值
    return: 异常标记数组
    """
    valid = np.isfinite(values)
    if valid.sum() < 3:
        return np.zeros(len(values), dtype=bool)

    filled = interpolate_nans(values)
    baseline = rolling_nanmedian(filled, baseline_window)
    residual = np.abs(filled - baseline)
    residual_valid = residual[valid]

    if residual_valid.size == 0:
        return np.zeros(len(values), dtype=bool)

    center = float(np.median(residual_valid))
    mad = float(np.median(np.abs(residual_valid - center)))
    robust_sigma = 1.4826 * mad
    threshold = max(center + mad_scale * robust_sigma, min_threshold)

    if not math.isfinite(threshold):
        return np.zeros(len(values), dtype=bool)

    return valid & (residual > threshold)


def smooth_metric(values: np.ndarray, anomaly_mask: np.ndarray, smooth_window: int) -> np.ndarray:
    """
    summary: 对单个指标做简单平滑
    param values: 输入指标序列
    param anomaly_mask: 异常标记数组
    param smooth_window: 平滑窗口大小
    return: 平滑后的序列
    """
    working = values.astype(np.float64).copy()
    working[anomaly_mask] = np.nan
    working = interpolate_nans(working)
    return moving_average(working, smooth_window)


def build_anomaly_reasons(metric_arrays: Dict[str, np.ndarray], metric_anomalies: Dict[str, np.ndarray]) -> List[str]:
    """
    summary: 为每一帧整理异常原因
    param metric_arrays: 原始指标字典
    param metric_anomalies: 各指标异常标记字典
    return: 每帧异常原因文本列表
    """
    reasons: List[str] = []
    frame_count = len(next(iter(metric_arrays.values())))

    for index in range(frame_count):
        parts: List[str] = []

        invalid_metrics = [name for name in METRIC_NAMES if not np.isfinite(metric_arrays[name][index])]
        if invalid_metrics:
            parts.append("invalid:" + ",".join(invalid_metrics))

        outlier_metrics = [name for name in METRIC_NAMES if metric_anomalies[name][index]]
        if outlier_metrics:
            parts.append("outlier:" + ",".join(outlier_metrics))

        reasons.append(" | ".join(parts))

    return reasons


def analyze_sequence(
    frame_ids: List[str],
    metric_arrays: Dict[str, np.ndarray],
    baseline_window: int,
    smooth_window: int,
    mad_scale: float,
    min_threshold: float,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, List[str]]:
    """
    summary: 对单个序列进行异常检测与平滑
    param frame_ids: 帧编号列表
    param metric_arrays: 原始指标字典
    param baseline_window: 异常检测基线窗口
    param smooth_window: 平滑窗口大小
    param mad_scale: MAD 阈值倍数
    param min_threshold: 最小异常阈值
    return: 平滑结果字典、总异常标记、异常原因列表
    """
    del frame_ids

    metric_anomalies: Dict[str, np.ndarray] = {}
    smoothed_metrics: Dict[str, np.ndarray] = {}

    for metric_name in METRIC_NAMES:
        anomalies = detect_metric_anomalies(
            values=metric_arrays[metric_name],
            baseline_window=baseline_window,
            mad_scale=mad_scale,
            min_threshold=min_threshold,
        )
        metric_anomalies[metric_name] = anomalies
        smoothed_metrics[metric_name] = smooth_metric(
            values=metric_arrays[metric_name],
            anomaly_mask=anomalies | ~np.isfinite(metric_arrays[metric_name]),
            smooth_window=smooth_window,
        )

    anomaly_reasons = build_anomaly_reasons(metric_arrays=metric_arrays, metric_anomalies=metric_anomalies)
    combined_anomaly = np.array([reason != "" for reason in anomaly_reasons], dtype=bool)
    return smoothed_metrics, combined_anomaly, anomaly_reasons


def save_analysis_csv(
    output_csv_path: Path,
    frame_ids: List[str],
    metric_arrays: Dict[str, np.ndarray],
    smoothed_metrics: Dict[str, np.ndarray],
    anomaly_flags: np.ndarray,
    anomaly_reasons: List[str],
) -> None:
    """
    summary: 保存带平滑结果和异常标记的分析 csv
    param output_csv_path: 输出 csv 路径
    param frame_ids: 帧编号列表
    param metric_arrays: 原始指标字典
    param smoothed_metrics: 平滑指标字典
    param anomaly_flags: 总异常标记
    param anomaly_reasons: 异常原因列表
    return: 无
    """
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "frame_id",
        "norm_dx",
        "norm_dy",
        "norm_radius",
        "smooth_norm_dx",
        "smooth_norm_dy",
        "smooth_norm_radius",
        "is_anomaly",
        "anomaly_reason",
    ]

    with open(output_csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for index, frame_id in enumerate(frame_ids):
            writer.writerow({
                "frame_id": frame_id,
                "norm_dx": metric_arrays["norm_dx"][index],
                "norm_dy": metric_arrays["norm_dy"][index],
                "norm_radius": metric_arrays["norm_radius"][index],
                "smooth_norm_dx": smoothed_metrics["norm_dx"][index],
                "smooth_norm_dy": smoothed_metrics["norm_dy"][index],
                "smooth_norm_radius": smoothed_metrics["norm_radius"][index],
                "is_anomaly": int(anomaly_flags[index]),
                "anomaly_reason": anomaly_reasons[index],
            })


def plot_sequence_analysis(
    sequence_name: str,
    output_plot_path: Path,
    frame_axis: np.ndarray,
    metric_arrays: Dict[str, np.ndarray],
    smoothed_metrics: Dict[str, np.ndarray],
    anomaly_flags: np.ndarray,
) -> None:
    """
    summary: 绘制原始与平滑后的指标对比图
    param sequence_name: 序列名称
    param output_plot_path: 输出图路径
    param frame_axis: 横轴数组
    param metric_arrays: 原始指标字典
    param smoothed_metrics: 平滑指标字典
    param anomaly_flags: 总异常标记
    return: 无
    """
    output_plot_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    figure.suptitle(f"{sequence_name} | Raw vs Smoothed | anomalies={int(anomaly_flags.sum())}", fontsize=14)

    for axis, metric_name in zip(axes, METRIC_NAMES):
        raw = metric_arrays[metric_name]
        smooth = smoothed_metrics[metric_name]
        valid_raw = np.isfinite(raw)
        anomaly_valid = anomaly_flags & valid_raw

        axis.plot(frame_axis[valid_raw], raw[valid_raw], color="#8a8f98", linewidth=1.2, alpha=0.8, label="raw")
        axis.plot(frame_axis[np.isfinite(smooth)], smooth[np.isfinite(smooth)], color="#0b6efd", linewidth=2.0, label="smoothed")

        if anomaly_valid.any():
            axis.scatter(
                frame_axis[anomaly_valid],
                raw[anomaly_valid],
                color="#d62728",
                marker="x",
                s=40,
                linewidths=1.5,
                label="anomaly",
                zorder=4,
            )

        for anomaly_x in frame_axis[anomaly_flags]:
            axis.axvline(anomaly_x, color="#ff6b6b", alpha=0.08, linewidth=1)

        axis.set_ylabel(metric_name)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper right")

    axes[-1].set_xlabel("frame")
    figure.tight_layout()
    figure.savefig(output_plot_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def process_all_sequences(
    input_root: str,
    output_root: str,
    sequence_glob: str = "S_*",
    csv_name: str = "geometry.csv",
    baseline_window: int = 7,
    smooth_window: int = 5,
    mad_scale: float = 4.0,
    min_threshold: float = 0.01,
) -> None:
    """
    summary: 批量分析所有序列 csv
    param input_root: 输入根目录
    param output_root: 输出根目录
    param sequence_glob: 序列匹配模式
    param csv_name: 目录模式下的 csv 文件名
    param baseline_window: 异常检测基线窗口
    param smooth_window: 平滑窗口大小
    param mad_scale: MAD 阈值倍数
    param min_threshold: 最小异常阈值
    return: 无
    """
    sequence_csvs = list_sequence_csvs(input_root=input_root, sequence_glob=sequence_glob, csv_name=csv_name)

    if len(sequence_csvs) == 0:
        raise RuntimeError(f"没有找到可分析的序列 csv: root={input_root}, glob={sequence_glob}")

    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    for index, (sequence_name, csv_path) in enumerate(sequence_csvs, start=1):
        print(f"[{index}/{len(sequence_csvs)}] 分析序列: {sequence_name}")

        frame_ids, metric_arrays = read_sequence_csv(csv_path)
        frame_axis = parse_frame_axis(frame_ids)

        smoothed_metrics, anomaly_flags, anomaly_reasons = analyze_sequence(
            frame_ids=frame_ids,
            metric_arrays=metric_arrays,
            baseline_window=baseline_window,
            smooth_window=smooth_window,
            mad_scale=mad_scale,
            min_threshold=min_threshold,
        )

        sequence_output_dir = output_root_path / sequence_name
        analysis_csv_path = sequence_output_dir / "sequence_analysis.csv"
        plot_path = sequence_output_dir / "sequence_analysis.png"

        save_analysis_csv(
            output_csv_path=analysis_csv_path,
            frame_ids=frame_ids,
            metric_arrays=metric_arrays,
            smoothed_metrics=smoothed_metrics,
            anomaly_flags=anomaly_flags,
            anomaly_reasons=anomaly_reasons,
        )

        plot_sequence_analysis(
            sequence_name=sequence_name,
            output_plot_path=plot_path,
            frame_axis=frame_axis,
            metric_arrays=metric_arrays,
            smoothed_metrics=smoothed_metrics,
            anomaly_flags=anomaly_flags,
        )

        print(f"  异常帧数量: {int(anomaly_flags.sum())}")
        print(f"  图像输出  : {plot_path}")
        print(f"  分析 CSV  : {analysis_csv_path}")

    print(f"\n全部序列分析完成，输出目录: {output_root_path}")


def parse_args() -> argparse.Namespace:
    """
    summary: 解析命令行参数
    param 无: 无
    return: 参数对象
    """
    parser = argparse.ArgumentParser(description="对每个 S_x 序列 csv 做曲线分析、异常标记和平滑对比")

    parser.add_argument("--input_root", type=str, required=True, help="输入根目录，可包含 S_x/geometry.csv 或 S_x.csv")
    parser.add_argument("--output_root", type=str, required=True, help="分析结果输出根目录")
    parser.add_argument("--sequence_glob", type=str, default="S_*", help="序列匹配模式，默认 S_*")
    parser.add_argument("--csv_name", type=str, default="geometry.csv", help="目录模式下的 csv 文件名，默认 geometry.csv")
    parser.add_argument("--baseline_window", type=int, default=7, help="异常检测基线窗口")
    parser.add_argument("--smooth_window", type=int, default=5, help="平滑窗口大小")
    parser.add_argument("--mad_scale", type=float, default=4.0, help="MAD 阈值倍数")
    parser.add_argument("--min_threshold", type=float, default=0.01, help="最小异常阈值")

    return parser.parse_args()


def main() -> None:
    """
    summary: 主函数，执行批量序列分析
    param 无: 无
    return: 无
    """
    args = parse_args()

    process_all_sequences(
        input_root=args.input_root,
        output_root=args.output_root,
        sequence_glob=args.sequence_glob,
        csv_name=args.csv_name,
        baseline_window=args.baseline_window,
        smooth_window=args.smooth_window,
        mad_scale=args.mad_scale,
        min_threshold=args.min_threshold,
    )


if __name__ == "__main__":
    main()


