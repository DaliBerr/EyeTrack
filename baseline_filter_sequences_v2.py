import os
import re
import math
import argparse
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def natural_key(text: str) -> List[Any]:
    """
    summary:
    param text: input
    return: list
    """
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def ensure_dir(dir_path: str) -> None:
    """
    summary: directory
    param dir_path: directorypath
    return: none
    """
    os.makedirs(dir_path, exist_ok=True)


def discover_sequence_csvs(input_path: str, csv_name: str = "geometry.csv") -> List[Tuple[str, str]]:
    """
    summary: sequence_outputs sequence geometry.csv
    param input_path: directory, sequencedirectory csv path
    param csv_name: sequencedirectory csv file
    return: list, (sequence_name, csv_path)
    """
    results: List[Tuple[str, str]] = []

    # 1: csv
    if os.path.isfile(input_path):
        if not input_path.lower().endswith(".csv"):
            raise ValueError(f"inputfile csv: {input_path}")

        parent_name = os.path.basename(os.path.dirname(input_path))
        seq_name = parent_name if parent_name else os.path.splitext(os.path.basename(input_path))[0]
        return [(seq_name, input_path)]

    # 2: sequencedirectory, S_37
    if os.path.isdir(input_path):
        direct_csv = os.path.join(input_path, csv_name)
        if os.path.isfile(direct_csv):
            seq_name = os.path.basename(os.path.normpath(input_path))
            return [(seq_name, direct_csv)]

        # 3: directory, sequence_outputs
        for name in os.listdir(input_path):
            subdir = os.path.join(input_path, name)
            if not os.path.isdir(subdir):
                continue

            csv_path = os.path.join(subdir, csv_name)
            if os.path.isfile(csv_path):
                results.append((name, csv_path))

        results.sort(key=lambda item: natural_key(item[0]))
        return results

    raise FileNotFoundError(f"inputpath: {input_path}")


def load_sequence_csv(csv_path: str) -> pd.DataFrame:
    """
    summary: read sequence csv
    param csv_path: csv filepath
    return: DataFrame
    """
    df = pd.read_csv(csv_path)

    required_cols = ["norm_dx", "norm_dy"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{csv_path} missing: {col}")

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
    summary:
    param series: inputsequence
    param window: window
    return: sequence
    """
    return series.rolling(window=window, center=True, min_periods=1).median()


def dilate_boolean_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """
    summary: mask
    param mask: input array
    param radius:
    return: array
    """
    if radius <= 0:
        return mask.copy()

    kernel = np.ones(2 * radius + 1, dtype=int)
    expanded = np.convolve(mask.astype(int), kernel, mode="same") > 0
    return expanded


def find_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    summary: array True
    param mask: input array
    return: list, (start, end),
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
    bad_pad: int
) -> Dict[str, np.ndarray]:
    """
    summary: anomaly
    param df: inputsequence DataFrame
    param radius_min: norm_radius minimumthreshold
    param radius_max: norm_radius maximumthreshold
    param area_ratio_min: pupil_iris_area_ratio minimumthreshold
    param area_ratio_max: pupil_iris_area_ratio maximumthreshold
    param dx_jump_thresh: norm_dx threshold
    param dy_jump_thresh: norm_dy threshold
    param dx_dev_thresh: norm_dx threshold
    param dy_dev_thresh: norm_dy threshold
    param local_window: window
    param bad_pad: anomaly
    return: anomalymask dict
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

    padded_bad = dilate_boolean_mask(core_bad, bad_pad)

    return {
        "bad_invalid": bad_invalid.fillna(False).to_numpy(),
        "bad_radius": bad_radius.fillna(False).to_numpy(),
        "bad_area": bad_area.fillna(False).to_numpy(),
        "bad_jump": bad_jump.fillna(False).to_numpy(),
        "bad_dev": bad_dev.fillna(False).to_numpy(),
        "core_bad": core_bad,
        "bad_frame": padded_bad,
    }


def fill_short_bad_runs(
    series: np.ndarray,
    bad_mask: np.ndarray,
    max_interp_gap: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    summary: anomaly, anomaly
    param series: sequence
    param bad_mask: anomaly mask
    param max_interp_gap: maximum anomaly
    return: sequence, mask, mask, array
    """
    values = series.astype(float).copy()
    interpolated_mask = np.zeros(len(values), dtype=bool)
    gap_mask = np.zeros(len(values), dtype=bool)
    run_length_arr = np.zeros(len(values), dtype=int)

    runs = find_runs(bad_mask)

    for start, end in runs:
        run_len = end - start + 1
        run_length_arr[start:end + 1] = run_len

        left_idx = start - 1
        right_idx = end + 1

        has_left = left_idx >= 0 and not bad_mask[left_idx] and np.isfinite(values[left_idx])
        has_right = right_idx < len(values) and not bad_mask[right_idx] and np.isfinite(values[right_idx])

        if run_len <= max_interp_gap and has_left and has_right:
            left_val = values[left_idx]
            right_val = values[right_idx]
            fill_count = run_len

            for k in range(fill_count):
                t = (k + 1) / (fill_count + 1)
                values[start + k] = (1.0 - t) * left_val + t * right_val
                interpolated_mask[start + k] = True
        else:
            values[start:end + 1] = np.nan
            gap_mask[start:end + 1] = True

    return values, interpolated_mask, gap_mask, run_length_arr


def median_then_ema_on_segment(
    segment: np.ndarray,
    median_window: int,
    ema_alpha: float
) -> np.ndarray:
    """
    summary: valid EMA
    param segment: valid
    param median_window: window
    param ema_alpha: EMA smoothing
    return: smoothing
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
    summary: valid smoothing
    param values: NaN sequence
    param median_window: window
    param ema_alpha: EMA smoothing
    return: smoothing sequence
    """
    output = values.astype(float).copy()
    valid_mask = np.isfinite(output)

    runs = find_runs(valid_mask)

    for start, end in runs:
        segment = output[start:end + 1]
        filtered = median_then_ema_on_segment(segment, median_window, ema_alpha)
        output[start:end + 1] = filtered

    return output


def build_clean_series(
    df: pd.DataFrame,
    bad_mask: np.ndarray,
    max_interp_gap: int,
    median_window: int,
    ema_alpha: float
) -> Dict[str, np.ndarray]:
    """
    summary: norm_dx, norm_dy, norm_radius
    param df: inputsequence DataFrame
    param bad_mask: anomalymask
    param max_interp_gap: maximumanomaly
    param median_window: window
    param ema_alpha: EMA smoothing
    return: dict
    """
    dx_raw = df["norm_dx"].to_numpy(dtype=float)
    dy_raw = df["norm_dy"].to_numpy(dtype=float)

    dx_stage1, dx_interp_mask, dx_gap_mask, dx_run_len = fill_short_bad_runs(dx_raw, bad_mask, max_interp_gap)
    dy_stage1, dy_interp_mask, dy_gap_mask, dy_run_len = fill_short_bad_runs(dy_raw, bad_mask, max_interp_gap)

    dx_clean = smooth_valid_segments(dx_stage1, median_window, ema_alpha)
    dy_clean = smooth_valid_segments(dy_stage1, median_window, ema_alpha)

    both_valid = np.isfinite(dx_clean) & np.isfinite(dy_clean)
    radius_clean = np.full(len(df), np.nan, dtype=float)
    radius_clean[both_valid] = np.sqrt(dx_clean[both_valid] ** 2 + dy_clean[both_valid] ** 2)

    interp_mask = dx_interp_mask | dy_interp_mask
    gap_mask = dx_gap_mask | dy_gap_mask
    run_len = np.maximum(dx_run_len, dy_run_len)

    return {
        "norm_dx_clean": dx_clean,
        "norm_dy_clean": dy_clean,
        "norm_radius_clean": radius_clean,
        "is_interpolated": interp_mask,
        "is_gap": gap_mask,
        "bad_run_length": run_len,
    }


def append_clean_columns(
    df: pd.DataFrame,
    detection: Dict[str, np.ndarray],
    cleaned: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """
    summary: anomaly DataFrame
    param df: DataFrame
    param detection: anomaly dict
    param cleaned: dict
    return: DataFrame
    """
    out = df.copy()

    out["bad_invalid"] = detection["bad_invalid"].astype(int)
    out["bad_radius"] = detection["bad_radius"].astype(int)
    out["bad_area"] = detection["bad_area"].astype(int)
    out["bad_jump"] = detection["bad_jump"].astype(int)
    out["bad_dev"] = detection["bad_dev"].astype(int)
    out["core_bad"] = detection["core_bad"].astype(int)
    out["bad_frame"] = detection["bad_frame"].astype(int)

    out["is_interpolated"] = cleaned["is_interpolated"].astype(int)
    out["is_gap"] = cleaned["is_gap"].astype(int)
    out["bad_run_length"] = cleaned["bad_run_length"].astype(int)

    out["norm_dx_clean"] = cleaned["norm_dx_clean"]
    out["norm_dy_clean"] = cleaned["norm_dy_clean"]
    out["norm_radius_clean"] = cleaned["norm_radius_clean"]

    return out


def shade_gap_runs(ax: plt.Axes, gap_mask: np.ndarray, color: str = "red", alpha: float = 0.08) -> None:
    """
    summary:
    param ax: matplotlib
    param gap_mask: mask
    param color:
    param alpha:
    return: none
    """
    for start, end in find_runs(gap_mask):
        ax.axvspan(start, end, color=color, alpha=alpha)


def plot_sequence_comparison(
    df: pd.DataFrame,
    seq_name: str,
    save_path: str
) -> None:
    """
    summary: raw vs cleaned
    param df: DataFrame
    param seq_name: sequence
    param save_path: savepath
    return: none
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

    anomaly_count = int(df["core_bad"].sum())
    gap_count = int(df["is_gap"].sum())
    interp_count = int(df["is_interpolated"].sum())

    axes[0].set_title(
        f"{seq_name} | Raw vs Cleaned | core_anomalies={anomaly_count} | gaps={gap_count} | interpolated={interp_count}"
    )
    axes[-1].set_xlabel("frame")

    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_summary_row(seq_name: str, df: pd.DataFrame, source_csv_path: str) -> Dict[str, Any]:
    """
    summary: sequence
    param seq_name: sequence
    param df: DataFrame
    param source_csv_path: csv path
    return: dict
    """
    clean_valid = df["norm_dx_clean"].notna() & df["norm_dy_clean"].notna()

    return {
        "sequence": seq_name,
        "source_csv": source_csv_path,
        "frames": len(df),
        "core_anomalies": int(df["core_bad"].sum()),
        "bad_frames_after_pad": int(df["bad_frame"].sum()),
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
    bad_pad: int,
    max_interp_gap: int,
    median_window: int,
    ema_alpha: float
) -> Dict[str, Any]:
    """
    summary: process sequence csv, output cleaned csv
    param seq_name: sequence
    param csv_path: input csv path
    param cleaned_csv_dir: csv directory
    param plot_dir: directory
    param radius_min: norm_radius minimumthreshold
    param radius_max: norm_radius maximumthreshold
    param area_ratio_min: pupil_iris_area_ratio minimumthreshold
    param area_ratio_max: pupil_iris_area_ratio maximumthreshold
    param dx_jump_thresh: norm_dx threshold
    param dy_jump_thresh: norm_dy threshold
    param dx_dev_thresh: norm_dx threshold
    param dy_dev_thresh: norm_dy threshold
    param local_window: window
    param bad_pad: anomaly
    param max_interp_gap: maximumanomaly
    param median_window: window
    param ema_alpha: EMA smoothing
    return: sequence
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
        bad_pad=bad_pad
    )

    cleaned = build_clean_series(
        df=df,
        bad_mask=detection["bad_frame"],
        max_interp_gap=max_interp_gap,
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
        f"core_anomalies={summary['core_anomalies']} | "
        f"interp={summary['interpolated_frames']} | "
        f"gaps={summary['gap_frames']} | "
        f"clean_valid={summary['clean_valid_frames']}"
    )

    return summary


def save_summary_csv(summary_rows: List[Dict[str, Any]], save_path: str) -> None:
    """
    summary: save sequence
    param summary_rows: dictlist
    param save_path: output csv path
    return: none
    """
    df = pd.DataFrame(summary_rows)
    df.to_csv(save_path, index=False)


def parse_args() -> argparse.Namespace:
    """
    summary: parseCLIarguments
    param none: none
    return: arguments
    """
    parser = argparse.ArgumentParser(description=" sequence_outputs/S_x/geometry.csv anomalyprocess, ")

    parser.add_argument("--input", type=str, required=True, help="sequence_outputs directory, S_x directory geometry.csv")
    parser.add_argument("--output_dir", type=str, required=True, help="outputdirectory")
    parser.add_argument("--csv_name", type=str, default="geometry.csv", help=" sequencedirectory csv file ")

    parser.add_argument("--radius_min", type=float, default=0.0, help="norm_radius minimumthreshold")
    parser.add_argument("--radius_max", type=float, default=0.8, help="norm_radius maximumthreshold")

    parser.add_argument("--area_ratio_min", type=float, default=0.02, help=" minimumthreshold")
    parser.add_argument("--area_ratio_max", type=float, default=0.20, help=" maximumthreshold")

    parser.add_argument("--dx_jump_thresh", type=float, default=0.025, help="norm_dx threshold")
    parser.add_argument("--dy_jump_thresh", type=float, default=0.18, help="norm_dy threshold")

    parser.add_argument("--dx_dev_thresh", type=float, default=0.025, help="norm_dx threshold")
    parser.add_argument("--dy_dev_thresh", type=float, default=0.18, help="norm_dy threshold")

    parser.add_argument("--local_window", type=int, default=7, help=" window")
    parser.add_argument("--bad_pad", type=int, default=1, help=" anomaly ")

    parser.add_argument("--max_interp_gap", type=int, default=2, help=" maximumanomaly ")
    parser.add_argument("--median_window", type=int, default=5, help=" window")
    parser.add_argument("--ema_alpha", type=float, default=0.35, help="EMA smoothing ")

    return parser.parse_args()


def main() -> None:
    """
    summary: main function, batchprocess sequence_outputs sequence csv
    param none: none
    return: none
    """
    args = parse_args()

    seq_items = discover_sequence_csvs(args.input, csv_name=args.csv_name)
    if len(seq_items) == 0:
        raise RuntimeError("No geometry.csv files were found.")

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
            bad_pad=args.bad_pad,
            max_interp_gap=args.max_interp_gap,
            median_window=args.median_window,
            ema_alpha=args.ema_alpha
        )
        summary_rows.append(summary)

    summary_csv_path = os.path.join(args.output_dir, "summary.csv")
    save_summary_csv(summary_rows, summary_csv_path)

    print("\nProcessing completed.")
    print(f"cleaned_csv directory: {cleaned_csv_dir}")
    print(f"plots directory: {plot_dir}")
    print(f"summary.csv: {summary_csv_path}")


if __name__ == "__main__":
    main()