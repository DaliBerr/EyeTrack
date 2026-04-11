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
    summary: 生成自然排序键
    param text: 输入字符串
    return: 可用于排序的键列表
    """
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def ensure_dir(dir_path: Optional[str]) -> None:
    """
    summary: 若目录路径非空则确保目录存在
    param dir_path: 目录路径
    return: 无
    """
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)


def list_prediction_files(pred_dir: str) -> List[str]:
    """
    summary: 列出预测目录中的所有分割结果文件
    param pred_dir: 分割结果目录
    return: 排序后的文件路径列表
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
    summary: 在指定目录中查找与 stem 同名的图像文件
    param folder: 目录路径
    param stem: 文件主名
    return: 若存在则返回文件路径，否则返回 None
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
    summary: 读取灰度图像
    param image_path: 图像路径
    return: uint8 灰度图，shape 为 HxW
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"无法读取图像文件: {image_path}")
    return image


def read_binary_mask(mask_path: str) -> np.ndarray:
    """
    summary: 读取二值 mask 并转为 0/1
    param mask_path: mask 路径
    return: uint8 二值图，shape 为 HxW，取值为 0 或 1
    """
    ext = os.path.splitext(mask_path)[1].lower()

    if ext == ".npy":
        mask = np.load(mask_path)
        if mask.ndim != 2:
            raise ValueError(f"mask 维度不是 2D: {mask_path}, shape={mask.shape}")
        mask = (mask > 0).astype(np.uint8)
        return mask

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"无法读取 mask 文件: {mask_path}")

    mask = (mask > 0).astype(np.uint8)
    return mask


def read_label_map(label_path: str) -> np.ndarray:
    """
    summary: 读取分割标签图，支持 npy 与普通灰度图
    param label_path: 标签图路径
    return: uint8 标签图，shape 为 HxW
    """
    ext = os.path.splitext(label_path)[1].lower()

    if ext == ".npy":
        label = np.load(label_path)
        if label.ndim != 2:
            raise ValueError(f"标签图维度不是 2D: {label_path}, shape={label.shape}")
        return label.astype(np.uint8)

    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    if label is None:
        raise FileNotFoundError(f"无法读取标签图文件: {label_path}")
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
    summary: 从预测标签图中提取 iris 与 pupil 的几何参数
    param pred_label_map: 输入预测标签图
    param valid_mask: 可选有效区域 mask
    param iris_class_id: 虹膜类别编号
    param pupil_class_id: 瞳孔类别编号
    param kernel_size: 形态学核大小
    param iris_min_area: iris 最小面积阈值
    param pupil_min_area: pupil 最小面积阈值
    return: 包含几何结果与中间 mask 的字典
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
    summary: 将椭圆对象转为 csv 行字段
    param prefix: 字段名前缀
    param ellipse: 椭圆对象
    return: 字典形式的字段
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
    summary: 将区域几何对象转为 csv 行字段
    param prefix: 字段名前缀
    param region: 区域几何对象
    return: 字典形式的字段
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
    summary: 将单帧几何结果整理为 csv 行
    param frame_id: 帧编号
    param result: 几何结果字典
    return: csv 行字段字典
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
    summary: 在图像上绘制椭圆
    param canvas: 输入彩色画布
    param ellipse: 椭圆对象
    param color: BGR 颜色
    param thickness: 线宽
    return: 绘制后的图像
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
    summary: 在图像上绘制中心点
    param canvas: 输入彩色画布
    param center_x: 中心点 x 坐标
    param center_y: 中心点 y 坐标
    param color: BGR 颜色
    param radius: 点半径
    return: 绘制后的图像
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
    summary: 在图像上叠加二值区域轮廓
    param canvas: 输入彩色画布
    param binary_mask: 二值区域
    param color: BGR 颜色
    param thickness: 线宽
    return: 绘制后的图像
    """
    output = canvas.copy()

    contour_img = (binary_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(contour_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(output, contours, -1, color, thickness)

    return output


def make_overlay_image(gray_image: np.ndarray, result: GazeFeatureResult, frame_id: str) -> np.ndarray:
    """
    summary: 生成几何参数叠加可视化图
    param gray_image: 原始灰度图
    param result: 几何结果字典
    param frame_id: 当前帧编号
    return: BGR 彩色叠加图
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
    summary: 保存二值图到 png 文件
    param mask: 输入二值图
    param save_path: 保存路径
    return: 无
    """
    cv2.imwrite(save_path, (mask * 255).astype(np.uint8))


def write_csv_rows(rows: List[Dict[str, Any]], csv_path: str) -> None:
    """
    summary: 将所有结果行写入 csv
    param rows: 结果行列表
    param csv_path: 输出 csv 路径
    return: 无
    """
    if len(rows) == 0:
        raise RuntimeError("没有可写入 csv 的结果行。")

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
    summary: 批量处理预测目录并导出几何参数
    param pred_dir: 分割结果目录
    param output_csv: 输出 csv 路径
    param image_dir: 原图目录，可为 None
    param mask_dir: 有效区域目录，可为 None
    param overlay_dir: 叠加图保存目录，可为 None
    param clean_mask_dir: 清理后二值图保存目录，可为 None
    param iris_class_id: iris 类别编号
    param pupil_class_id: pupil 类别编号
    param kernel_size: 形态学核大小
    param iris_min_area: iris 最小面积阈值
    param pupil_min_area: pupil 最小面积阈值
    param save_overlay: 是否保存叠加图
    param save_clean_masks: 是否保存清理后的二值图
    return: 无
    """
    pred_files = list_prediction_files(pred_dir)

    if len(pred_files) == 0:
        raise RuntimeError(f"在目录中没有找到分割结果文件: {pred_dir}")

    if save_overlay:
        ensure_dir(overlay_dir)

    iris_mask_save_dir = None
    pupil_mask_save_dir = None
    if save_clean_masks:
        if clean_mask_dir is None:
            raise ValueError("save_clean_masks=True 时，clean_mask_dir 不能为空。")
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
    print(f"\n已完成，csv 已保存到: {output_csv}")

    if save_overlay and overlay_dir is not None:
        print(f"叠加图目录: {overlay_dir}")

    if save_clean_masks and clean_mask_dir is not None:
        print(f"清理后二值图目录: {clean_mask_dir}")


def parse_args() -> argparse.Namespace:
    """
    summary: 解析命令行参数
    param 无: 无
    return: 参数对象
    """
    parser = argparse.ArgumentParser(description="从分割结果中提取 iris/pupil 几何参数")

    parser.add_argument("--pred_dir", type=str, required=True, help="分割结果目录，支持 npy/png")
    parser.add_argument("--output_csv", type=str, required=True, help="输出 csv 路径")

    parser.add_argument("--image_dir", type=str, default=None, help="原图目录，可选")
    parser.add_argument("--mask_dir", type=str, default=None, help="有效区域 mask 目录，可选")

    parser.add_argument("--overlay_dir", type=str, default=None, help="叠加图输出目录")
    parser.add_argument("--clean_mask_dir", type=str, default=None, help="清理后二值图输出目录")

    parser.add_argument("--save_overlay", action="store_true", help="是否保存叠加图")
    parser.add_argument("--save_clean_masks", action="store_true", help="是否保存清理后二值图")

    parser.add_argument("--iris_class_id", type=int, default=2, help="iris 类别编号")
    parser.add_argument("--pupil_class_id", type=int, default=3, help="pupil 类别编号")

    parser.add_argument("--kernel_size", type=int, default=3, help="形态学核大小")
    parser.add_argument("--iris_min_area", type=int, default=100, help="iris 最小面积阈值")
    parser.add_argument("--pupil_min_area", type=int, default=20, help="pupil 最小面积阈值")

    return parser.parse_args()


def main() -> None:
    """
    summary: 主函数，批量处理分割结果并导出几何参数
    param 无: 无
    return: 无
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
