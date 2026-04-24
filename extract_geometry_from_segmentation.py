import os
import re
import csv
import argparse
from typing import Optional, List, Tuple, Dict, Any

import cv2
import numpy as np
from eyetrack.gaze import (
    EllipseResult,
    GazeFeatureResult,
    RegionGeometry,
    extract_gaze_features_from_label_map,
)


def natural_key(text: str) -> List[Any]:
    """
    summary:
    param text: input
    return: list
    """
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def ensure_dir(dir_path: Optional[str]) -> None:
    """
    summary: directorypath directory
    param dir_path: directorypath
    return: none
    """
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)


def list_prediction_files(pred_dir: str) -> List[str]:
    """
    summary: predictiondirectory file
    param pred_dir: directory
    return: filepathlist
    """
    valid_exts = {".npy", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    file_list = []

    for name in os.listdir(pred_dir):
        path = os.path.join(pred_dir, name)
        if os.path.isfile(path):
            ext = os.path.splitext(name)[1].lower()
            if ext in valid_exts:
                file_list.append(path)

    file_list.sort(key=lambda p: natural_key(os.path.basename(p)))
    return file_list


def find_matching_file_by_stem(folder: Optional[str], stem: str) -> Optional[str]:
    """
    summary: directory stem imagefile
    param folder: directorypath
    param stem: file
    return: returnfilepath, return None
    """
    if folder is None:
        return None

    candidates = [
        f"{stem}.png",
        f"{stem}.jpg",
        f"{stem}.jpeg",
        f"{stem}.bmp",
        f"{stem}.tif",
        f"{stem}.tiff",
        f"{stem}.npy",
    ]

    for name in candidates:
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path

    return None


def read_gray_image(image_path: str) -> np.ndarray:
    """
    summary: read image
    param image_path: imagepath
    return: uint8, shape HxW
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"unable toreadimagefile: {image_path}")
    return image


def read_binary_mask(mask_path: str) -> np.ndarray:
    """
    summary: read mask 0/1
    param mask_path: mask path
    return: uint8, shape HxW, 0 1
    """
    ext = os.path.splitext(mask_path)[1].lower()

    if ext == ".npy":
        mask = np.load(mask_path)
        if mask.ndim != 2:
            raise ValueError(f"mask 2D: {mask_path}, shape={mask.shape}")
        mask = (mask > 0).astype(np.uint8)
        return mask

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"unable toread mask file: {mask_path}")

    mask = (mask > 0).astype(np.uint8)
    return mask


def read_label_map(label_path: str) -> np.ndarray:
    """
    summary: read label, supports npy
    param label_path: label path
    return: uint8 label, shape HxW
    """
    ext = os.path.splitext(label_path)[1].lower()

    if ext == ".npy":
        label = np.load(label_path)
        if label.ndim != 2:
            raise ValueError(f"label 2D: {label_path}, shape={label.shape}")
        return label.astype(np.uint8)

    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    if label is None:
        raise FileNotFoundError(f"unable toreadlabel file: {label_path}")
    return label.astype(np.uint8)


def extract_eye_geometry_from_label_map(
    pred_label_map: np.ndarray,
    valid_mask: Optional[np.ndarray],
    iris_class_id: int,
    pupil_class_id: int,
    kernel_size: int,
    iris_min_area: int,
    pupil_min_area: int
) -> GazeFeatureResult:
    """
    summary: predictionlabel iris pupil arguments
    param pred_label_map: inputpredictionlabel
    param valid_mask: optionalvalid mask
    param iris_class_id: irisclass
    param pupil_class_id: pupilclass
    param kernel_size:
    param iris_min_area: iris minimum threshold
    param pupil_min_area: pupil minimum threshold
    return: mask dict
    """
    return extract_gaze_features_from_label_map(
        pred_label_map=pred_label_map,
        valid_mask=valid_mask,
        iris_class_id=iris_class_id,
        pupil_class_id=pupil_class_id,
        kernel_size=kernel_size,
        iris_min_area=iris_min_area,
        pupil_min_area=pupil_min_area,
    )


def ellipse_to_row(prefix: str, ellipse: Optional[EllipseResult]) -> Dict[str, Any]:
    """
    summary: csv
    param prefix:
    param ellipse:
    return: dict
    """
    if ellipse is None:
        return {
            f"{prefix}_ellipse_cx": None,
            f"{prefix}_ellipse_cy": None,
            f"{prefix}_major_axis": None,
            f"{prefix}_minor_axis": None,
            f"{prefix}_angle_deg": None,
        }

    return {
        f"{prefix}_ellipse_cx": ellipse.center_x,
        f"{prefix}_ellipse_cy": ellipse.center_y,
        f"{prefix}_major_axis": ellipse.major_axis,
        f"{prefix}_minor_axis": ellipse.minor_axis,
        f"{prefix}_angle_deg": ellipse.angle_deg,
    }


def region_to_row(prefix: str, region: RegionGeometry) -> Dict[str, Any]:
    """
    summary: csv
    param prefix:
    param region:
    return: dict
    """
    row = {
        f"{prefix}_area": region.area,
        f"{prefix}_center_x": region.center_x,
        f"{prefix}_center_y": region.center_y,
        f"{prefix}_valid": int(region.area > 0),
    }
    row.update(ellipse_to_row(prefix, region.ellipse))
    return row


def build_csv_row(frame_id: str, result: GazeFeatureResult) -> Dict[str, Any]:
    """
    summary: csv
    param frame_id:
    param result: dict
    return: csv dict
    """
    row = {"frame_id": frame_id}

    row.update(region_to_row("iris", result.iris_geometry))
    row.update(region_to_row("pupil", result.pupil_geometry))

    row["offset_dx"] = result.offset_dx
    row["offset_dy"] = result.offset_dy
    row["local_dx"] = result.local_dx
    row["local_dy"] = result.local_dy
    row["norm_dx"] = result.norm_dx
    row["norm_dy"] = result.norm_dy
    row["norm_radius"] = result.norm_radius
    row["pupil_iris_area_ratio"] = result.pupil_iris_area_ratio

    return row


def draw_ellipse(canvas: np.ndarray, ellipse: Optional[EllipseResult], color: Tuple[int, int, int], thickness: int) -> np.ndarray:
    """
    summary: image
    param canvas: input
    param ellipse:
    param color: BGR
    param thickness:
    return: image
    """
    output = canvas.copy()

    if ellipse is None:
        return output

    center = (int(round(ellipse.center_x)), int(round(ellipse.center_y)))
    axes = (
        int(round(ellipse.major_axis / 2.0)),
        int(round(ellipse.minor_axis / 2.0))
    )

    cv2.ellipse(
        output,
        center=center,
        axes=axes,
        angle=ellipse.angle_deg,
        startAngle=0,
        endAngle=360,
        color=color,
        thickness=thickness
    )
    return output


def draw_center(canvas: np.ndarray, center_x: Optional[float], center_y: Optional[float], color: Tuple[int, int, int], radius: int) -> np.ndarray:
    """
    summary: image
    param canvas: input
    param center_x: x
    param center_y: y
    param color: BGR
    param radius:
    return: image
    """
    output = canvas.copy()

    if center_x is None or center_y is None:
        return output

    cv2.circle(
        output,
        center=(int(round(center_x)), int(round(center_y))),
        radius=radius,
        color=color,
        thickness=-1
    )
    return output


def overlay_mask_contour(canvas: np.ndarray, binary_mask: np.ndarray, color: Tuple[int, int, int], thickness: int) -> np.ndarray:
    """
    summary: image
    param canvas: input
    param binary_mask:
    param color: BGR
    param thickness:
    return: image
    """
    output = canvas.copy()

    contour_img = (binary_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(contour_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(output, contours, -1, color, thickness)

    return output


def make_overlay_image(gray_image: np.ndarray, result: GazeFeatureResult, frame_id: str) -> np.ndarray:
    """
    summary: arguments
    param gray_image:
    param result: dict
    param frame_id: current
    return: BGR
    """
    canvas = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    iris_color = (0, 255, 0)
    pupil_color = (0, 0, 255)

    if result.iris_mask is not None:
        canvas = overlay_mask_contour(canvas, result.iris_mask, iris_color, 2)
    if result.pupil_mask is not None:
        canvas = overlay_mask_contour(canvas, result.pupil_mask, pupil_color, 2)

    canvas = draw_ellipse(canvas, result.iris_geometry.ellipse, iris_color, 2)
    canvas = draw_ellipse(canvas, result.pupil_geometry.ellipse, pupil_color, 2)

    canvas = draw_center(canvas, result.iris_geometry.center_x, result.iris_geometry.center_y, iris_color, 3)
    canvas = draw_center(canvas, result.pupil_geometry.center_x, result.pupil_geometry.center_y, pupil_color, 3)

    text_lines = [
        f"id={frame_id}",
        f"iris_area={result.iris_geometry.area}",
        f"pupil_area={result.pupil_geometry.area}",
        f"dx={result.offset_dx}",
        f"dy={result.offset_dy}",
        f"norm_dx={result.norm_dx}",
        f"norm_dy={result.norm_dy}",
        f"norm_r={result.norm_radius}",
    ]

    y = 22
    for line in text_lines:
        cv2.putText(
            canvas,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            1,
            cv2.LINE_AA
        )
        y += 22

    return canvas


def save_binary_mask(mask: np.ndarray, save_path: str) -> None:
    """
    summary: save png file
    param mask: input
    param save_path: savepath
    return: none
    """
    cv2.imwrite(save_path, (mask * 255).astype(np.uint8))


def write_csv_rows(rows: List[Dict[str, Any]], csv_path: str) -> None:
    """
    summary: csv
    param rows: list
    param csv_path: output csv path
    return: none
    """
    if len(rows) == 0:
        raise RuntimeError("No result rows available to write into CSV.")

    ensure_dir(os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".")

    fieldnames = list(rows[0].keys())

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_prediction_directory(
    pred_dir: str,
    output_csv: str,
    image_dir: Optional[str],
    mask_dir: Optional[str],
    overlay_dir: Optional[str],
    clean_mask_dir: Optional[str],
    iris_class_id: int,
    pupil_class_id: int,
    kernel_size: int,
    iris_min_area: int,
    pupil_min_area: int,
    save_overlay: bool,
    save_clean_masks: bool
) -> None:
    """
    summary: batchprocesspredictiondirectory arguments
    param pred_dir: directory
    param output_csv: output csv path
    param image_dir: directory, None
    param mask_dir: valid directory, None
    param overlay_dir: savedirectory, None
    param clean_mask_dir: savedirectory, None
    param iris_class_id: iris class
    param pupil_class_id: pupil class
    param kernel_size:
    param iris_min_area: iris minimum threshold
    param pupil_min_area: pupil minimum threshold
    param save_overlay: save
    param save_clean_masks: save
    return: none
    """
    pred_files = list_prediction_files(pred_dir)

    if len(pred_files) == 0:
        raise RuntimeError(f" directory file: {pred_dir}")

    if save_overlay:
        ensure_dir(overlay_dir)

    iris_mask_save_dir = None
    pupil_mask_save_dir = None
    if save_clean_masks:
        if clean_mask_dir is None:
            raise ValueError("save_clean_masks=True, clean_mask_dir.")
        iris_mask_save_dir = os.path.join(clean_mask_dir, "iris")
        pupil_mask_save_dir = os.path.join(clean_mask_dir, "pupil")
        ensure_dir(iris_mask_save_dir)
        ensure_dir(pupil_mask_save_dir)

    rows = []

    for idx, pred_path in enumerate(pred_files, start=1):
        frame_id = os.path.splitext(os.path.basename(pred_path))[0]

        pred_label_map = read_label_map(pred_path)

        image_path = find_matching_file_by_stem(image_dir, frame_id)
        mask_path = find_matching_file_by_stem(mask_dir, frame_id)

        valid_mask = None
        if mask_path is not None:
            valid_mask = read_binary_mask(mask_path)

        result = extract_eye_geometry_from_label_map(
            pred_label_map=pred_label_map,
            valid_mask=valid_mask,
            iris_class_id=iris_class_id,
            pupil_class_id=pupil_class_id,
            kernel_size=kernel_size,
            iris_min_area=iris_min_area,
            pupil_min_area=pupil_min_area
        )

        row = build_csv_row(frame_id, result)
        rows.append(row)

        if save_overlay:
            if image_path is not None:
                gray_image = read_gray_image(image_path)
            else:
                gray_image = np.zeros_like(pred_label_map, dtype=np.uint8)

            overlay = make_overlay_image(
                gray_image=gray_image,
                result=result,
                frame_id=frame_id
            )

            overlay_path = os.path.join(overlay_dir, f"{frame_id}_overlay.png")
            cv2.imwrite(overlay_path, overlay)

        if save_clean_masks:
            iris_path = os.path.join(iris_mask_save_dir, f"{frame_id}.png")
            pupil_path = os.path.join(pupil_mask_save_dir, f"{frame_id}.png")
            if result.iris_mask is not None:
                save_binary_mask(result.iris_mask, iris_path)
            if result.pupil_mask is not None:
                save_binary_mask(result.pupil_mask, pupil_path)

        print(
            f"[{idx}/{len(pred_files)}] "
            f"{frame_id} | "
            f"iris_area={result.iris_geometry.area} | "
            f"pupil_area={result.pupil_geometry.area} | "
            f"norm_dx={result.norm_dx} | "
            f"norm_dy={result.norm_dy}"
        )

    write_csv_rows(rows, output_csv)
    print(f"\nCompleted, csv save: {output_csv}")

    if save_overlay and overlay_dir is not None:
        print(f" directory: {overlay_dir}")

    if save_clean_masks and clean_mask_dir is not None:
        print(f" directory: {clean_mask_dir}")


def parse_args() -> argparse.Namespace:
    """
    summary: parseCLIarguments
    param none: none
    return: arguments
    """
    parser = argparse.ArgumentParser(description=" iris/pupil arguments")

    parser.add_argument("--pred_dir", type=str, required=True, help=" directory, supports npy/png")
    parser.add_argument("--output_csv", type=str, required=True, help="output csv path")

    parser.add_argument("--image_dir", type=str, default=None, help=" directory, optional")
    parser.add_argument("--mask_dir", type=str, default=None, help="valid mask directory, optional")

    parser.add_argument("--overlay_dir", type=str, default=None, help=" outputdirectory")
    parser.add_argument("--clean_mask_dir", type=str, default=None, help=" outputdirectory")

    parser.add_argument("--save_overlay", action="store_true", help=" save ")
    parser.add_argument("--save_clean_masks", action="store_true", help=" save ")

    parser.add_argument("--iris_class_id", type=int, default=2, help="iris class ")
    parser.add_argument("--pupil_class_id", type=int, default=3, help="pupil class ")

    parser.add_argument("--kernel_size", type=int, default=3, help=" ")
    parser.add_argument("--iris_min_area", type=int, default=100, help="iris minimum threshold")
    parser.add_argument("--pupil_min_area", type=int, default=20, help="pupil minimum threshold")

    return parser.parse_args()


def main() -> None:
    """
    summary: main function, batchprocess arguments
    param none: none
    return: none
    """
    args = parse_args()

    process_prediction_directory(
        pred_dir=args.pred_dir,
        output_csv=args.output_csv,
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        overlay_dir=args.overlay_dir,
        clean_mask_dir=args.clean_mask_dir,
        iris_class_id=args.iris_class_id,
        pupil_class_id=args.pupil_class_id,
        kernel_size=args.kernel_size,
        iris_min_area=args.iris_min_area,
        pupil_min_area=args.pupil_min_area,
        save_overlay=args.save_overlay,
        save_clean_masks=args.save_clean_masks
    )


if __name__ == "__main__":
    main()
