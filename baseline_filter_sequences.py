import argparse
import csv
import fnmatch
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


METRIC_NAMES = ["norm_dx", "norm_dy", "norm_radius"]


def natural_key(text: str) -> List[Any]:
    """
    summary:
    param text: input
    return: list
    """
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def parse_optional_float(text: str) -> float:
    """
    summary: parse, return NaN
    param text: input
    return: NaN
    """
    if text is None:
        return float("nan")

    value = text.strip()
    if value == "":
        return float("nan")

    return float(value)


def parse_frame_axis(frame_ids: List[str]) -> np.ndarray:
    """
    summary: frame_id list
    param frame_ids: list
    return: array
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
    summary: process sequence csv
    param input_root: input directory
    param sequence_glob: sequence
    param csv_name: directory csv file
    return: (sequence, csv path) list
    """
    root_path = Path(input_root)

    if not root_path.exists():
        raise FileNotFoundError(f"not foundinput directory: {root_path}")

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
    summary: read sequence csv
    param csv_path: csv filepath
    return: frame_id list dict
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
    summary:
    param values: inputarray
    param window_size: window
    return: smoothingbaselinearray
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
    summary: NaN
    param values: inputarray
    return: array
    """
    output = values.astype(np.float64).copy()
    valid = np.isfinite(output)

    if not valid.any():
        return output

    if valid.sum() == 1:
        output[~valid] = output[valid][0]
        return output

    valid_index = np.flatnonzero(valid)
    full_index = np.arange(len(output))
    output[~valid] = np.interp(full_index[~valid], valid_index, output[valid])
    return output


def median_filter(values: np.ndarray, window_size: int) -> np.ndarray:
    """
    summary: sequence
    param values: inputarray
    param window_size: window
    return:
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
            output[index] = np.median(window[finite])

    return output


def ema_filter(values: np.ndarray, alpha: float) -> np.ndarray:
    """
    summary: sequence
    param values: inputarray
    param alpha: EMA
    return: EMA
    """
    if len(values) == 0:
        return values.copy()

    output = np.full_like(values, np.nan, dtype=np.float64)
    valid = np.isfinite(values)

    if not valid.any():
        return output

    first_valid_index = int(np.flatnonzero(valid)[0])
    output[first_valid_index] = values[first_valid_index]

    for index in range(first_valid_index + 1, len(values)):
        current_value = values[index]
        previous_value = output[index - 1]

        if not np.isfinite(previous_value):
            output[index] = current_value
        elif np.isfinite(current_value):
            output[index] = alpha * current_value + (1.0 - alpha) * previous_value
        else:
            output[index] = previous_value

    for index in range(first_valid_index - 1, -1, -1):
        output[index] = output[index + 1]

    return output


def detect_metric_anomalies(
    values: np.ndarray,
    baseline_window: int,
    mad_scale: float,
    min_threshold: float,
) -> np.ndarray:
    """
    summary: anomaly
    param values: input sequence
    param baseline_window: baselinewindow
    param mad_scale: MAD threshold
    param min_threshold: minimumthreshold
    return: anomaly array
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


def build_anomaly_reasons(metric_arrays: Dict[str, np.ndarray], metric_anomalies: Dict[str, np.ndarray]) -> List[str]:
    """
    summary: anomaly
    param metric_arrays: dict
    param metric_anomalies: anomaly dict
    return: anomaly list
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


def clean_metric(
    values: np.ndarray,
    anomaly_mask: np.ndarray,
    median_window: int,
    ema_alpha: float,
) -> Dict[str, np.ndarray]:
    """
    summary:, EMA
    param values: sequence
    param anomaly_mask: anomaly array
    param median_window: window
    param ema_alpha: EMA
    return: dict
    """
    working = values.astype(np.float64).copy()
    working[anomaly_mask] = np.nan
    working[~np.isfinite(working)] = np.nan

    interpolated = interpolate_nans(working)
    medianed = median_filter(interpolated, median_window)
    emaed = ema_filter(medianed, ema_alpha)

    return {
        "interpolated": interpolated,
        "median": medianed,
        "cleaned": emaed,
    }


def save_cleaned_csv(
    output_csv_path: Path,
    frame_ids: List[str],
    metric_arrays: Dict[str, np.ndarray],
    cleaned_results: Dict[str, Dict[str, np.ndarray]],
    anomaly_flags: np.ndarray,
    anomaly_reasons: List[str],
) -> None:
    """
    summary: save baseline csv
    param output_csv_path: output csv path
    param frame_ids: list
    param metric_arrays: dict
    param cleaned_results:
    param anomaly_flags: anomaly
    param anomaly_reasons: anomaly list
    return: none
    """
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "frame_id",
        "norm_dx",
        "norm_dy",
        "norm_radius",
        "interp_norm_dx",
        "interp_norm_dy",
        "interp_norm_radius",
        "median_norm_dx",
        "median_norm_dy",
        "median_norm_radius",
        "cleaned_norm_dx",
        "cleaned_norm_dy",
        "cleaned_norm_radius",
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
                "interp_norm_dx": cleaned_results["norm_dx"]["interpolated"][index],
                "interp_norm_dy": cleaned_results["norm_dy"]["interpolated"][index],
                "interp_norm_radius": cleaned_results["norm_radius"]["interpolated"][index],
                "median_norm_dx": cleaned_results["norm_dx"]["median"][index],
                "median_norm_dy": cleaned_results["norm_dy"]["median"][index],
                "median_norm_radius": cleaned_results["norm_radius"]["median"][index],
                "cleaned_norm_dx": cleaned_results["norm_dx"]["cleaned"][index],
                "cleaned_norm_dy": cleaned_results["norm_dy"]["cleaned"][index],
                "cleaned_norm_radius": cleaned_results["norm_radius"]["cleaned"][index],
                "is_anomaly": int(anomaly_flags[index]),
                "anomaly_reason": anomaly_reasons[index],
            })


def plot_raw_vs_cleaned(
    sequence_name: str,
    output_plot_path: Path,
    frame_axis: np.ndarray,
    metric_arrays: Dict[str, np.ndarray],
    cleaned_results: Dict[str, Dict[str, np.ndarray]],
    anomaly_flags: np.ndarray,
) -> None:
    """
    summary:
    param sequence_name: sequence
    param output_plot_path: output path
    param frame_axis: array
    param metric_arrays: dict
    param cleaned_results:
    param anomaly_flags: anomaly
    return: none
    """
    output_plot_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    figure.suptitle(f"{sequence_name} | Raw vs Cleaned | anomalies={int(anomaly_flags.sum())}", fontsize=14)

    for axis, metric_name in zip(axes, METRIC_NAMES):
        raw_values = metric_arrays[metric_name]
        cleaned_values = cleaned_results[metric_name]["cleaned"]
        raw_valid = np.isfinite(raw_values)
        cleaned_valid = np.isfinite(cleaned_values)
        anomaly_valid = anomaly_flags & raw_valid

        axis.plot(frame_axis[raw_valid], raw_values[raw_valid], color="#8a8f98", linewidth=1.2, alpha=0.85, label="raw")
        axis.plot(frame_axis[cleaned_valid], cleaned_values[cleaned_valid], color="#0b6efd", linewidth=2.0, label="cleaned")

        if anomaly_valid.any():
            axis.scatter(
                frame_axis[anomaly_valid],
                raw_values[anomaly_valid],
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
    median_window: int = 5,
    ema_alpha: float = 0.35,
    mad_scale: float = 4.0,
    min_threshold: float = 0.01,
) -> None:
    """
    summary: batch sequence baseline
    param input_root: input directory
    param output_root: output directory
    param sequence_glob: sequence
    param csv_name: directory csv file
    param baseline_window: anomaly baselinewindow
    param median_window: window
    param ema_alpha: EMA
    param mad_scale: MAD threshold
    param min_threshold: minimumanomalythreshold
    return: none
    """
    sequence_csvs = list_sequence_csvs(input_root=input_root, sequence_glob=sequence_glob, csv_name=csv_name)

    if len(sequence_csvs) == 0:
        raise RuntimeError(f" process sequence csv: root={input_root}, glob={sequence_glob}")

    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    for index, (sequence_name, csv_path) in enumerate(sequence_csvs, start=1):
        print(f"[{index}/{len(sequence_csvs)}] baseline processsequence: {sequence_name}")

        frame_ids, metric_arrays = read_sequence_csv(csv_path)
        frame_axis = parse_frame_axis(frame_ids)

        metric_anomalies: Dict[str, np.ndarray] = {}
        cleaned_results: Dict[str, Dict[str, np.ndarray]] = {}

        for metric_name in METRIC_NAMES:
            anomalies = detect_metric_anomalies(
                values=metric_arrays[metric_name],
                baseline_window=baseline_window,
                mad_scale=mad_scale,
                min_threshold=min_threshold,
            )
            invalid_mask = ~np.isfinite(metric_arrays[metric_name])
            metric_anomalies[metric_name] = anomalies | invalid_mask
            cleaned_results[metric_name] = clean_metric(
                values=metric_arrays[metric_name],
                anomaly_mask=metric_anomalies[metric_name],
                median_window=median_window,
                ema_alpha=ema_alpha,
            )

        anomaly_reasons = build_anomaly_reasons(metric_arrays=metric_arrays, metric_anomalies=metric_anomalies)
        anomaly_flags = np.array([reason != "" for reason in anomaly_reasons], dtype=bool)

        sequence_output_dir = output_root_path / sequence_name
        cleaned_csv_path = sequence_output_dir / "baseline_cleaned.csv"
        plot_path = sequence_output_dir / "baseline_raw_vs_cleaned.png"

        save_cleaned_csv(
            output_csv_path=cleaned_csv_path,
            frame_ids=frame_ids,
            metric_arrays=metric_arrays,
            cleaned_results=cleaned_results,
            anomaly_flags=anomaly_flags,
            anomaly_reasons=anomaly_reasons,
        )

        plot_raw_vs_cleaned(
            sequence_name=sequence_name,
            output_plot_path=plot_path,
            frame_axis=frame_axis,
            metric_arrays=metric_arrays,
            cleaned_results=cleaned_results,
            anomaly_flags=anomaly_flags,
        )

        print(f" anomaly count: {int(anomaly_flags.sum())}")
        print(f" output: {plot_path}")
        print(f"  cleaned csv: {cleaned_csv_path}")

    print(f"\n baseline completed, outputdirectory: {output_root_path}")


def parse_args() -> argparse.Namespace:
    """
    summary: parseCLIarguments
    param none: none
    return: arguments
    """
    parser = argparse.ArgumentParser(description=" S_x.csv anomaly,, EMA baseline")

    parser.add_argument("--input_root", type=str, required=True, help="input directory, S_x/geometry.csv S_x.csv")
    parser.add_argument("--output_root", type=str, required=True, help="baseline output directory")
    parser.add_argument("--sequence_glob", type=str, default="S_*", help="sequence, default S_*")
    parser.add_argument("--csv_name", type=str, default="geometry.csv", help="directory csv file, default geometry.csv")
    parser.add_argument("--baseline_window", type=int, default=7, help="anomaly baselinewindow")
    parser.add_argument("--median_window", type=int, default=5, help=" window ")
    parser.add_argument("--ema_alpha", type=float, default=0.35, help="EMA, 0~1")
    parser.add_argument("--mad_scale", type=float, default=4.0, help="MAD threshold ")
    parser.add_argument("--min_threshold", type=float, default=0.01, help="minimumanomalythreshold")

    return parser.parse_args()


def main() -> None:
    """
    summary: main function, baseline
    param none: none
    return: none
    """
    args = parse_args()

    process_all_sequences(
        input_root=args.input_root,
        output_root=args.output_root,
        sequence_glob=args.sequence_glob,
        csv_name=args.csv_name,
        baseline_window=args.baseline_window,
        median_window=args.median_window,
        ema_alpha=args.ema_alpha,
        mad_scale=args.mad_scale,
        min_threshold=args.min_threshold,
    )


if __name__ == "__main__":
    main()
