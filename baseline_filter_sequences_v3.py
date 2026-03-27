import os
import re
import argparse
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def natural_key(text: str) -> List[Any]:
    """
    summary: 生成自然排序键
    param text: 输入字符串
    return: 可用于自然排序的键列表
    """
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def ensure_dir(dir_path: str) -> None:
    """
    summary: 确保目录存在
    param dir_path: 目录路径
    return: 无
    """
    os.makedirs(dir_path, exist_ok=True)


def discover_sequence_csvs(input_path: str, csv_name: str = "geometry.csv") -> List[Tuple[str, str]]:
    """
    summary: 发现 sequence_outputs 结构下的各序列 geometry.csv
    param input_path: 根目录、单个序列目录或单个 csv 路径
    param csv_name: 序列目录中的 csv 文件名
    return: 列表，每项为 (sequence_name, csv_path)
    """
    results: List[Tuple[str, str]] = []

    if os.path.isfile(input_path):
        if not input_path.lower().endswith(".csv"):
            raise ValueError(f"输入文件不是 csv: {input_path}")
        parent_name = os.path.basename(os.path.dirname(input_path))
        seq_name = parent_name if parent_name else os.path.splitext(os.path.basename(input_path))[0]
        return [(seq_name, input_path)]

    if os.path.isdir(input_path):
        direct_csv = os.path.join(input_path, csv_name)
        if os.path.isfile(direct_csv):
            seq_name = os.path.basename(os.path.normpath(input_path))
            return [(seq_name, direct_csv)]

        for name in os.listdir(input_path):
            subdir = os.path.join(input_path, name)
            if not os.path.isdir(subdir):
                continue

            csv_path = os.path.join(subdir, csv_name)
            if os.path.isfile(csv_path):
                results.append((name, csv_path))

        results.sort(key=lambda item: natural_key(item[0]))
        return results

    raise FileNotFoundError(f"输入路径不存在: {input_path}")


def load_sequence_csv(csv_path: str) -> pd.DataFrame:
    """
    summary: 读取单个序列 csv 并做基础列检查
    param csv_path: csv 文件路径
    return: 包含基础字段的 DataFrame
    """
    df = pd.read_csv(csv_path)

    required_cols = ["norm_dx", "norm_dy"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{csv_path} 缺少必要列: {col}")

    if "frame" not in df.columns:
        df["frame"] = np.arange(len(df), dtype=int)

    if "norm_radius" not in df.columns:
        df["norm_radius"] = np.sqrt(df["norm_dx"] ** 2 + df["norm_dy"] ** 2)

    if "iris_valid" not in df.columns:
        df["iris_valid"] = 1

    if "pupil_valid" not in df.columns:
        df["pupil_valid"] = 1

    if "pupil_iris_area_ratio" not in df.columns:
        df["pupil_iris_area_ratio"] = np.nan

    return df


def rolling_median(series: pd.Series, window: int) -> pd.Series:
    """
    summary: 计算中心滑动中位数
    param series: 输入序列
    param window: 窗口大小
    return: 中位数序列
    """
    return series.rolling(window=window, center=True, min_periods=1).median()


def dilate_boolean_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """
    summary: 对布尔掩码做一维膨胀
    param mask: 输入布尔数组
    param radius: 膨胀半径
    return: 膨胀后的布尔数组
    """
    if radius <= 0:
        return mask.copy()

    kernel = np.ones(2 * radius + 1, dtype=int)
    expanded = np.convolve(mask.astype(int), kernel, mode="same") > 0
    return expanded


def find_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    summary: 找出布尔数组中为 True 的连续区间
    param mask: 输入布尔数组
    return: 连续区间列表，每项为 (start, end)，包含端点
    """
    runs: List[Tuple[int, int]] = []
    n = len(mask)
    i = 0

    while i < n:
        if mask[i]:
            start = i
            while i + 1 < n and mask[i + 1]:
                i += 1
            end = i
            runs.append((start, end))
        i += 1

    return runs


def detect_bad_frames(
    df: pd.DataFrame,
    radius_min: float,
    radius_max: float,
    area_ratio_min: float,
    area_ratio_max: float,
    dx_jump_thresh: float,
    dy_jump_thresh: float,
    dx_dev_thresh: float,
    dy_dev_thresh: float,
    local_window: int,
    anomaly_display_pad: int
) -> Dict[str, np.ndarray]:
    """
    summary: 综合多个规则做逐帧核心异常检测
    param df: 输入序列 DataFrame
    param radius_min: norm_radius 最小阈值
    param radius_max: norm_radius 最大阈值
    param area_ratio_min: 面积比最小阈值
    param area_ratio_max: 面积比最大阈值
    param dx_jump_thresh: norm_dx 单步跳变阈值
    param dy_jump_thresh: norm_dy 单步跳变阈值
    param dx_dev_thresh: norm_dx 相对局部中位数偏差阈值
    param dy_dev_thresh: norm_dy 相对局部中位数偏差阈值
    param local_window: 局部中位数窗口
    param anomaly_display_pad: 仅用于显示的异常扩张帧数
    return: 异常检测结果字典
    """
    dx = df["norm_dx"].astype(float)
    dy = df["norm_dy"].astype(float)
    radius = df["norm_radius"].astype(float)
    area_ratio = df["pupil_iris_area_ratio"].astype(float)

    bad_invalid = (df["iris_valid"].fillna(0).astype(int) == 0) | (df["pupil_valid"].fillna(0).astype(int) == 0)
    bad_radius = (radius < radius_min) | (radius > radius_max)

    area_has_value = area_ratio.notna()
    bad_area = pd.Series(False, index=df.index)
    bad_area[area_has_value] = (area_ratio[area_has_value] < area_ratio_min) | (area_ratio[area_has_value] > area_ratio_max)

    dx_diff = dx.diff().abs()
    dy_diff = dy.diff().abs()

    bad_jump_dx = (dx_diff > dx_jump_thresh) | (dx_diff.shift(-1) > dx_jump_thresh)
    bad_jump_dy = (dy_diff > dy_jump_thresh) | (dy_diff.shift(-1) > dy_jump_thresh)
    bad_jump = bad_jump_dx.fillna(False) | bad_jump_dy.fillna(False)

    dx_med = rolling_median(dx, local_window)
    dy_med = rolling_median(dy, local_window)

    bad_dev_dx = (dx - dx_med).abs() > dx_dev_thresh
    bad_dev_dy = (dy - dy_med).abs() > dy_dev_thresh
    bad_dev = bad_dev_dx.fillna(False) | bad_dev_dy.fillna(False)

    core_bad = (
        bad_invalid.fillna(False).to_numpy() |
        bad_radius.fillna(False).to_numpy() |
        bad_area.fillna(False).to_numpy() |
        bad_jump.fillna(False).to_numpy() |
        bad_dev.fillna(False).to_numpy()
    )

    display_bad = dilate_boolean_mask(core_bad, anomaly_display_pad)

    return {
        "bad_invalid": bad_invalid.fillna(False).to_numpy(),
        "bad_radius": bad_radius.fillna(False).to_numpy(),
        "bad_area": bad_area.fillna(False).to_numpy(),
        "bad_jump": bad_jump.fillna(False).to_numpy(),
        "bad_dev": bad_dev.fillna(False).to_numpy(),
        "core_bad": core_bad,
        "display_bad": display_bad,
    }


def split_core_bad_runs(core_bad: np.ndarray, max_interp_gap: int) -> Dict[str, np.ndarray]:
    """
    summary: 用核心异常掩码划分短异常段与长异常段
    param core_bad: 核心异常布尔掩码
    param max_interp_gap: 允许插值的最大连续异常长度
    return: 包含短异常、长异常与连续长度的字典
    """
    short_bad = np.zeros_like(core_bad, dtype=bool)
    long_bad = np.zeros_like(core_bad, dtype=bool)
    run_length = np.zeros(len(core_bad), dtype=int)

    for start, end in find_runs(core_bad):
        length = end - start + 1
        run_length[start:end + 1] = length

        if length <= max_interp_gap:
            short_bad[start:end + 1] = True
        else:
            long_bad[start:end + 1] = True

    return {
        "short_bad": short_bad,
        "long_bad": long_bad,
        "run_length": run_length,
    }


def interpolate_short_runs(series: np.ndarray, short_bad_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    summary: 仅对短异常段做线性插值
    param series: 原始一维数值序列
    param short_bad_mask: 短异常段布尔掩码
    return: 插值后序列与插值掩码
    """
    values = series.astype(float).copy()
    interpolated_mask = np.zeros(len(values), dtype=bool)

    for start, end in find_runs(short_bad_mask):
        left_idx = start - 1
        right_idx = end + 1

        has_left = left_idx >= 0 and not short_bad_mask[left_idx] and np.isfinite(values[left_idx])
        has_right = right_idx < len(values) and not short_bad_mask[right_idx] and np.isfinite(values[right_idx])

        if has_left and has_right:
            left_val = values[left_idx]
            right_val = values[right_idx]
            count = end - start + 1

            for k in range(count):
                t = (k + 1) / (count + 1)
                values[start + k] = (1.0 - t) * left_val + t * right_val
                interpolated_mask[start + k] = True
        else:
            values[start:end + 1] = np.nan

    return values, interpolated_mask


def apply_long_gap_mask(series: np.ndarray, long_bad_mask: np.ndarray) -> np.ndarray:
    """
    summary: 将长异常段置为 NaN
    param series: 输入序列
    param long_bad_mask: 长异常段布尔掩码
    return: 应用长异常空洞后的序列
    """
    output = series.astype(float).copy()
    output[long_bad_mask] = np.nan
    return output


def median_then_ema_on_segment(
    segment: np.ndarray,
    median_window: int,
    ema_alpha: float
) -> np.ndarray:
    """
    summary: 对单个连续有效段先做中值滤波再做 EMA
    param segment: 单段有效数据
    param median_window: 中值滤波窗口
    param ema_alpha: EMA 平滑系数
    return: 平滑后的单段数据
    """
    seg = pd.Series(segment.astype(float))

    if median_window >= 3 and len(seg) >= 3:
        if median_window % 2 == 0:
            median_window += 1
        seg = seg.rolling(window=median_window, center=True, min_periods=1).median()

    seg = seg.ewm(alpha=ema_alpha, adjust=False).mean()
    return seg.to_numpy()


def smooth_valid_segments(
    values: np.ndarray,
    median_window: int,
    ema_alpha: float
) -> np.ndarray:
    """
    summary: 仅对连续有效段分别做平滑
    param values: 包含 NaN 的一维数值序列
    param median_window: 中值滤波窗口
    param ema_alpha: EMA 平滑系数
    return: 分段平滑后的序列
    """
    output = values.astype(float).copy()
    valid_mask = np.isfinite(output)

    for start, end in find_runs(valid_mask):
        segment = output[start:end + 1]
        filtered = median_then_ema_on_segment(segment, median_window, ema_alpha)
        output[start:end + 1] = filtered

    return output


def build_clean_series(
    df: pd.DataFrame,
    core_bad: np.ndarray,
    max_interp_gap: int,
    smooth_gap_pad: int,
    median_window: int,
    ema_alpha: float
) -> Dict[str, np.ndarray]:
    """
    summary: 构建清洗后的 norm_dx、norm_dy、norm_radius
    param df: 输入序列 DataFrame
    param core_bad: 核心异常掩码
    param max_interp_gap: 允许插值的最大异常段长度
    param smooth_gap_pad: 仅对长异常段向两侧扩张的平滑保护帧数
    param median_window: 中值滤波窗口
    param ema_alpha: EMA 平滑系数
    return: 清洗结果字典
    """
    dx_raw = df["norm_dx"].to_numpy(dtype=float)
    dy_raw = df["norm_dy"].to_numpy(dtype=float)

    split_result = split_core_bad_runs(core_bad, max_interp_gap)
    short_bad = split_result["short_bad"]
    long_bad = split_result["long_bad"]
    run_length = split_result["run_length"]

    dx_stage1, dx_interp_mask = interpolate_short_runs(dx_raw, short_bad)
    dy_stage1, dy_interp_mask = interpolate_short_runs(dy_raw, short_bad)

    dx_stage2 = apply_long_gap_mask(dx_stage1, long_bad)
    dy_stage2 = apply_long_gap_mask(dy_stage1, long_bad)

    protected_gap = dilate_boolean_mask(long_bad, smooth_gap_pad)

    dx_for_smooth = dx_stage2.copy()
    dy_for_smooth = dy_stage2.copy()
    dx_for_smooth[protected_gap] = np.nan
    dy_for_smooth[protected_gap] = np.nan

    dx_clean = smooth_valid_segments(dx_for_smooth, median_window, ema_alpha)
    dy_clean = smooth_valid_segments(dy_for_smooth, median_window, ema_alpha)

    both_valid = np.isfinite(dx_clean) & np.isfinite(dy_clean)
    radius_clean = np.full(len(df), np.nan, dtype=float)
    radius_clean[both_valid] = np.sqrt(dx_clean[both_valid] ** 2 + dy_clean[both_valid] ** 2)

    return {
        "norm_dx_clean": dx_clean,
        "norm_dy_clean": dy_clean,
        "norm_radius_clean": radius_clean,
        "short_bad": short_bad,
        "long_bad": long_bad,
        "is_interpolated": dx_interp_mask | dy_interp_mask,
        "is_gap": protected_gap,
        "bad_run_length": run_length,
    }


def append_clean_columns(
    df: pd.DataFrame,
    detection: Dict[str, np.ndarray],
    cleaned: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """
    summary: 将异常检测与清洗结果追加到 DataFrame
    param df: 原始 DataFrame
    param detection: 异常检测结果字典
    param cleaned: 清洗结果字典
    return: 追加结果后的新 DataFrame
    """
    out = df.copy()

    out["bad_invalid"] = detection["bad_invalid"].astype(int)
    out["bad_radius"] = detection["bad_radius"].astype(int)
    out["bad_area"] = detection["bad_area"].astype(int)
    out["bad_jump"] = detection["bad_jump"].astype(int)
    out["bad_dev"] = detection["bad_dev"].astype(int)

    out["core_bad"] = detection["core_bad"].astype(int)
    out["display_bad"] = detection["display_bad"].astype(int)

    out["short_bad"] = cleaned["short_bad"].astype(int)
    out["long_bad"] = cleaned["long_bad"].astype(int)
    out["is_interpolated"] = cleaned["is_interpolated"].astype(int)
    out["is_gap"] = cleaned["is_gap"].astype(int)
    out["bad_run_length"] = cleaned["bad_run_length"].astype(int)

    out["norm_dx_clean"] = cleaned["norm_dx_clean"]
    out["norm_dy_clean"] = cleaned["norm_dy_clean"]
    out["norm_radius_clean"] = cleaned["norm_radius_clean"]

    return out


def shade_gap_runs(ax: plt.Axes, gap_mask: np.ndarray, color: str = "red", alpha: float = 0.08) -> None:
    """
    summary: 在图上用半透明色块标出长空洞区间
    param ax: matplotlib 坐标轴
    param gap_mask: 空洞掩码
    param color: 颜色
    param alpha: 透明度
    return: 无
    """
    for start, end in find_runs(gap_mask):
        ax.axvspan(start, end, color=color, alpha=alpha)


def plot_sequence_comparison(
    df: pd.DataFrame,
    seq_name: str,
    save_path: str
) -> None:
    """
    summary: 绘制 raw vs cleaned 对比图
    param df: 包含清洗结果的 DataFrame
    param seq_name: 序列名称
    param save_path: 图片保存路径
    return: 无
    """
    frame = df["frame"].to_numpy()
    core_bad = df["core_bad"].astype(bool).to_numpy()
    gap_mask = df["is_gap"].astype(bool).to_numpy()

    fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True)

    plot_items = [
        ("norm_dx", "norm_dx_clean", "norm_dx"),
        ("norm_dy", "norm_dy_clean", "norm_dy"),
        ("norm_radius", "norm_radius_clean", "norm_radius"),
    ]

    for ax, (raw_col, clean_col, ylabel) in zip(axes, plot_items):
        raw_values = df[raw_col].to_numpy(dtype=float)
        clean_values = df[clean_col].to_numpy(dtype=float)

        ax.plot(frame, raw_values, color="gray", alpha=0.75, linewidth=1.4, label="raw")
        ax.plot(frame, clean_values, color="#0066ff", linewidth=2.0, label="cleaned")

        if np.any(core_bad):
            ax.scatter(
                frame[core_bad],
                raw_values[core_bad],
                marker="x",
                color="red",
                s=42,
                linewidths=1.4,
                label="core_anomaly"
            )

        shade_gap_runs(ax, gap_mask, color="red", alpha=0.06)

        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")

    core_anomaly_count = int(df["core_bad"].sum())
    gap_count = int(df["is_gap"].sum())
    interp_count = int(df["is_interpolated"].sum())

    axes[0].set_title(
        f"{seq_name} | Raw vs Cleaned | core_anomalies={core_anomaly_count} | gaps={gap_count} | interpolated={interp_count}"
    )
    axes[-1].set_xlabel("frame")

    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_summary_row(seq_name: str, df: pd.DataFrame, source_csv_path: str) -> Dict[str, Any]:
    """
    summary: 生成单个序列的摘要统计
    param seq_name: 序列名称
    param df: 包含清洗结果的 DataFrame
    param source_csv_path: 原始 csv 路径
    return: 摘要字典
    """
    clean_valid = df["norm_dx_clean"].notna() & df["norm_dy_clean"].notna()

    return {
        "sequence": seq_name,
        "source_csv": source_csv_path,
        "frames": len(df),
        "core_anomalies": int(df["core_bad"].sum()),
        "short_bad_frames": int(df["short_bad"].sum()),
        "long_bad_frames": int(df["long_bad"].sum()),
        "interpolated_frames": int(df["is_interpolated"].sum()),
        "gap_frames": int(df["is_gap"].sum()),
        "clean_valid_frames": int(clean_valid.sum()),
        "clean_valid_ratio": float(clean_valid.mean()) if len(df) > 0 else 0.0,
    }


def process_one_sequence(
    seq_name: str,
    csv_path: str,
    cleaned_csv_dir: str,
    plot_dir: str,
    radius_min: float,
    radius_max: float,
    area_ratio_min: float,
    area_ratio_max: float,
    dx_jump_thresh: float,
    dy_jump_thresh: float,
    dx_dev_thresh: float,
    dy_dev_thresh: float,
    local_window: int,
    anomaly_display_pad: int,
    max_interp_gap: int,
    smooth_gap_pad: int,
    median_window: int,
    ema_alpha: float
) -> Dict[str, Any]:
    """
    summary: 处理单个序列 csv，输出 cleaned csv 与对比图
    param seq_name: 序列名称
    param csv_path: 输入 csv 路径
    param cleaned_csv_dir: 清洗后 csv 目录
    param plot_dir: 对比图目录
    param radius_min: norm_radius 最小阈值
    param radius_max: norm_radius 最大阈值
    param area_ratio_min: 面积比最小阈值
    param area_ratio_max: 面积比最大阈值
    param dx_jump_thresh: norm_dx 单步跳变阈值
    param dy_jump_thresh: norm_dy 单步跳变阈值
    param dx_dev_thresh: norm_dx 局部偏差阈值
    param dy_dev_thresh: norm_dy 局部偏差阈值
    param local_window: 局部中位数窗口
    param anomaly_display_pad: 仅用于显示的异常扩张帧数
    param max_interp_gap: 允许插值的最大异常段长度
    param smooth_gap_pad: 长异常段平滑保护扩张帧数
    param median_window: 中值滤波窗口
    param ema_alpha: EMA 平滑系数
    return: 单个序列的摘要统计
    """
    df = load_sequence_csv(csv_path)

    detection = detect_bad_frames(
        df=df,
        radius_min=radius_min,
        radius_max=radius_max,
        area_ratio_min=area_ratio_min,
        area_ratio_max=area_ratio_max,
        dx_jump_thresh=dx_jump_thresh,
        dy_jump_thresh=dy_jump_thresh,
        dx_dev_thresh=dx_dev_thresh,
        dy_dev_thresh=dy_dev_thresh,
        local_window=local_window,
        anomaly_display_pad=anomaly_display_pad
    )

    cleaned = build_clean_series(
        df=df,
        core_bad=detection["core_bad"],
        max_interp_gap=max_interp_gap,
        smooth_gap_pad=smooth_gap_pad,
        median_window=median_window,
        ema_alpha=ema_alpha
    )

    result_df = append_clean_columns(df, detection, cleaned)

    cleaned_csv_path = os.path.join(cleaned_csv_dir, f"{seq_name}_cleaned.csv")
    plot_path = os.path.join(plot_dir, f"{seq_name}_raw_vs_cleaned.png")

    result_df.to_csv(cleaned_csv_path, index=False)
    plot_sequence_comparison(result_df, seq_name, plot_path)

    summary = build_summary_row(seq_name, result_df, csv_path)

    print(
        f"{seq_name} | "
        f"frames={summary['frames']} | "
        f"core={summary['core_anomalies']} | "
        f"short={summary['short_bad_frames']} | "
        f"long={summary['long_bad_frames']} | "
        f"interp={summary['interpolated_frames']} | "
        f"gaps={summary['gap_frames']} | "
        f"clean_valid={summary['clean_valid_frames']}"
    )

    return summary


def save_summary_csv(summary_rows: List[Dict[str, Any]], save_path: str) -> None:
    """
    summary: 保存所有序列的摘要统计
    param summary_rows: 摘要字典列表
    param save_path: 输出 csv 路径
    return: 无
    """
    df = pd.DataFrame(summary_rows)
    df.to_csv(save_path, index=False)


def parse_args() -> argparse.Namespace:
    """
    summary: 解析命令行参数
    param 无: 无
    return: 参数对象
    """
    parser = argparse.ArgumentParser(description="对 sequence_outputs/S_x/geometry.csv 做异常处理、短段插值与分段滤波")

    parser.add_argument("--input", type=str, required=True, help="sequence_outputs 根目录、单个 S_x 目录或单个 geometry.csv")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--csv_name", type=str, default="geometry.csv", help="每个序列目录中的 csv 文件名")

    parser.add_argument("--radius_min", type=float, default=0.0, help="norm_radius 最小阈值")
    parser.add_argument("--radius_max", type=float, default=0.8, help="norm_radius 最大阈值")

    parser.add_argument("--area_ratio_min", type=float, default=0.02, help="面积比最小阈值")
    parser.add_argument("--area_ratio_max", type=float, default=0.20, help="面积比最大阈值")

    parser.add_argument("--dx_jump_thresh", type=float, default=0.025, help="norm_dx 单步跳变阈值")
    parser.add_argument("--dy_jump_thresh", type=float, default=0.18, help="norm_dy 单步跳变阈值")

    parser.add_argument("--dx_dev_thresh", type=float, default=0.025, help="norm_dx 局部偏差阈值")
    parser.add_argument("--dy_dev_thresh", type=float, default=0.18, help="norm_dy 局部偏差阈值")

    parser.add_argument("--local_window", type=int, default=7, help="局部中位数窗口")
    parser.add_argument("--anomaly_display_pad", type=int, default=1, help="仅用于显示的异常扩张帧数")

    parser.add_argument("--max_interp_gap", type=int, default=2, help="允许插值的最大核心异常段长度")
    parser.add_argument("--smooth_gap_pad", type=int, default=1, help="仅对长异常段做平滑保护扩张的帧数")

    parser.add_argument("--median_window", type=int, default=5, help="中值滤波窗口")
    parser.add_argument("--ema_alpha", type=float, default=0.35, help="EMA 平滑系数")

    return parser.parse_args()


def main() -> None:
    """
    summary: 主函数，批量处理 sequence_outputs 结构下的序列 csv
    param 无: 无
    return: 无
    """
    args = parse_args()

    seq_items = discover_sequence_csvs(args.input, csv_name=args.csv_name)
    if len(seq_items) == 0:
        raise RuntimeError("没有找到任何 geometry.csv。")

    cleaned_csv_dir = os.path.join(args.output_dir, "cleaned_csv")
    plot_dir = os.path.join(args.output_dir, "plots")
    ensure_dir(cleaned_csv_dir)
    ensure_dir(plot_dir)

    summary_rows = []

    for seq_name, csv_path in seq_items:
        summary = process_one_sequence(
            seq_name=seq_name,
            csv_path=csv_path,
            cleaned_csv_dir=cleaned_csv_dir,
            plot_dir=plot_dir,
            radius_min=args.radius_min,
            radius_max=args.radius_max,
            area_ratio_min=args.area_ratio_min,
            area_ratio_max=args.area_ratio_max,
            dx_jump_thresh=args.dx_jump_thresh,
            dy_jump_thresh=args.dy_jump_thresh,
            dx_dev_thresh=args.dx_dev_thresh,
            dy_dev_thresh=args.dy_dev_thresh,
            local_window=args.local_window,
            anomaly_display_pad=args.anomaly_display_pad,
            max_interp_gap=args.max_interp_gap,
            smooth_gap_pad=args.smooth_gap_pad,
            median_window=args.median_window,
            ema_alpha=args.ema_alpha
        )
        summary_rows.append(summary)

    summary_csv_path = os.path.join(args.output_dir, "summary.csv")
    save_summary_csv(summary_rows, summary_csv_path)

    print("\n处理完成。")
    print(f"cleaned_csv 目录: {cleaned_csv_dir}")
    print(f"plots 目录: {plot_dir}")
    print(f"summary.csv: {summary_csv_path}")


if __name__ == "__main__":
    main()