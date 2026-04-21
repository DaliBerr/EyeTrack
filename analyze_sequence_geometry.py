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
    summary: sequence csv
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

    valid_index = np.flatnonzero(valid)
    full_index = np.arange(len(output))
    output[~valid] = np.interp(full_index[~valid], valid_index, output[valid])
    return output


def moving_average(values: np.ndarray, window_size: int) -> np.ndarray:
    """
    summary:
    param values: inputarray
    param window_size: window
    return: smoothing array
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


def smooth_metric(values: np.ndarray, anomaly_mask: np.ndarray, smooth_window: int) -> np.ndarray:
    """
    summary: smoothing
    param values: input sequence
    param anomaly_mask: anomaly array
    param smooth_window: smoothingwindow
    return: smoothing sequence
    """
    working = values.astype(np.float64).copy()
    working[anomaly_mask] = np.nan
    working = interpolate_nans(working)
    return moving_average(working, smooth_window)


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


def analyze_sequence(
    frame_ids: List[str],
    metric_arrays: Dict[str, np.ndarray],
    baseline_window: int,
    smooth_window: int,
    mad_scale: float,
    min_threshold: float,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, List[str]]:
    """
    summary: sequence anomaly smoothing
    param frame_ids: list
    param metric_arrays: dict
    param baseline_window: anomaly baselinewindow
    param smooth_window: smoothingwindow
    param mad_scale: MAD threshold
    param min_threshold: minimumanomalythreshold
    return: smoothing dict, anomaly, anomaly list
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
    summary: save smoothing anomaly csv
    param output_csv_path: output csv path
    param frame_ids: list
    param metric_arrays: dict
    param smoothed_metrics: smoothing dict
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
    summary: smoothing
    param sequence_name: sequence
    param output_plot_path: output path
    param frame_axis: array
    param metric_arrays: dict
    param smoothed_metrics: smoothing dict
    param anomaly_flags: anomaly
    return: none
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
    summary: batch sequence csv
    param input_root: input directory
    param output_root: output directory
    param sequence_glob: sequence
    param csv_name: directory csv file
    param baseline_window: anomaly baselinewindow
    param smooth_window: smoothingwindow
    param mad_scale: MAD threshold
    param min_threshold: minimumanomalythreshold
    return: none
    """
    sequence_csvs = list_sequence_csvs(input_root=input_root, sequence_glob=sequence_glob, csv_name=csv_name)

    if len(sequence_csvs) == 0:
        raise RuntimeError(f" sequence csv: root={input_root}, glob={sequence_glob}")

    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    for index, (sequence_name, csv_path) in enumerate(sequence_csvs, start=1):
        print(f"[{index}/{len(sequence_csvs)}] sequence: {sequence_name}")

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

        print(f" anomaly count: {int(anomaly_flags.sum())}")
        print(f"  imageoutput  : {plot_path}")
        print(f" CSV: {analysis_csv_path}")

    print(f"\n sequence completed, outputdirectory: {output_root_path}")


def parse_args() -> argparse.Namespace:
    """
    summary: parseCLIarguments
    param none: none
    return: arguments
    """
    parser = argparse.ArgumentParser(description=" S_x sequence csv, anomaly smoothing ")

    parser.add_argument("--input_root", type=str, required=True, help="input directory, S_x/geometry.csv S_x.csv")
    parser.add_argument("--output_root", type=str, required=True, help=" output directory")
    parser.add_argument("--sequence_glob", type=str, default="S_*", help="sequence, default S_*")
    parser.add_argument("--csv_name", type=str, default="geometry.csv", help="directory csv file, default geometry.csv")
    parser.add_argument("--baseline_window", type=int, default=7, help="anomaly baselinewindow")
    parser.add_argument("--smooth_window", type=int, default=5, help="smoothingwindow ")
    parser.add_argument("--mad_scale", type=float, default=4.0, help="MAD threshold ")
    parser.add_argument("--min_threshold", type=float, default=0.01, help="minimumanomalythreshold")

    return parser.parse_args()


def main() -> None:
    """
    summary: main function, batchsequence
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
        smooth_window=args.smooth_window,
        mad_scale=args.mad_scale,
        min_threshold=args.min_threshold,
    )


if __name__ == "__main__":
    main()


