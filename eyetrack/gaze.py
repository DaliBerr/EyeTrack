from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import cv2
import numpy as np


@dataclass
class EllipseResult:
    center_x: float
    center_y: float
    major_axis: float
    minor_axis: float
    angle_deg: float


@dataclass
class RegionGeometry:
    area: int
    center_x: Optional[float]
    center_y: Optional[float]
    ellipse: Optional[EllipseResult]


@dataclass
class GazeFeatureResult:
    iris_mask: Optional[np.ndarray]
    pupil_mask: Optional[np.ndarray]
    iris_geometry: RegionGeometry
    pupil_geometry: RegionGeometry
    outer_boundary_geometry: RegionGeometry
    offset_dx: Optional[float]
    offset_dy: Optional[float]
    local_dx: Optional[float]
    local_dy: Optional[float]
    norm_dx: Optional[float]
    norm_dy: Optional[float]
    norm_radius: Optional[float]
    iris_feature_x: Optional[float]
    iris_feature_y: Optional[float]
    iris_outer_feature_x: Optional[float]
    iris_outer_feature_y: Optional[float]
    iris_outer_valid: bool
    iris_only_valid: bool
    pupil_iris_area_ratio: Optional[float]
    invalid_reasons: tuple[str, ...]
    valid: bool


@dataclass(frozen=True)
class CalibrationPoint:
    name: str
    u: float
    v: float


@dataclass(frozen=True)
class CalibrationSample:
    point_name: str
    target_u: float
    target_v: float
    feature_dx: float
    feature_dy: float
    weight: float = 1.0


@dataclass
class CalibrationModel:
    matrix: np.ndarray
    point_names: tuple[str, ...]
    source_features: np.ndarray
    target_points: np.ndarray
    mapping_type: str = "affine"
    static_boundary_ellipse: Optional[EllipseResult] = None


@dataclass
class CalibrationSession:
    points: list[CalibrationPoint]
    settle_ms: int
    capture_ms: int
    min_valid_frames: int
    state: str = "idle"
    point_index: int = 0
    phase_started_at_ms: Optional[float] = None
    current_point_samples: list[tuple[float, float, float]] = field(default_factory=list)
    boundary_samples: list[tuple[float, float, float, float, float]] = field(default_factory=list)
    collected_samples: list[CalibrationSample] = field(default_factory=list)
    model: Optional[CalibrationModel] = None
    failure_reason: Optional[str] = None

    @property
    def current_point(self) -> Optional[CalibrationPoint]:
        if 0 <= self.point_index < len(self.points):
            return self.points[self.point_index]
        return None

    @property
    def is_active(self) -> bool:
        return self.state in {"settling", "capturing"}

    @property
    def is_calibrated(self) -> bool:
        return self.state == "completed" and self.model is not None

    @property
    def calibration_step(self) -> str:
        point = self.current_point
        if point is None:
            return "done"
        return f"{self.point_index + 1}/{len(self.points)} {point.name}"


def build_empty_region_geometry() -> RegionGeometry:
    return RegionGeometry(area=0, center_x=None, center_y=None, ellipse=None)


def build_empty_gaze_feature_result() -> GazeFeatureResult:
    empty_region = build_empty_region_geometry()
    return GazeFeatureResult(
        iris_mask=None,
        pupil_mask=None,
        iris_geometry=empty_region,
        pupil_geometry=empty_region,
        outer_boundary_geometry=empty_region,
        offset_dx=None,
        offset_dy=None,
        local_dx=None,
        local_dy=None,
        norm_dx=None,
        norm_dy=None,
        norm_radius=None,
        iris_feature_x=None,
        iris_feature_y=None,
        iris_outer_feature_x=None,
        iris_outer_feature_y=None,
        iris_outer_valid=False,
        iris_only_valid=False,
        pupil_iris_area_ratio=None,
        invalid_reasons=tuple(),
        valid=False,
    )


def apply_valid_mask(label_map: np.ndarray, valid_mask: Optional[np.ndarray]) -> np.ndarray:
    output = label_map.copy()
    if valid_mask is not None:
        output[valid_mask == 0] = 0
    return output


def extract_class_binary_mask(label_map: np.ndarray, class_id: int) -> np.ndarray:
    return (label_map == class_id).astype(np.uint8)


def extract_non_background_mask(label_map: np.ndarray) -> np.ndarray:
    return (label_map > 0).astype(np.uint8)


def keep_largest_connected_component(binary_mask: np.ndarray) -> np.ndarray:
    if binary_mask.sum() == 0:
        return binary_mask.copy()

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels <= 1:
        return binary_mask.copy()

    largest_label = 1
    largest_area = stats[1, cv2.CC_STAT_AREA]
    for label_id in range(2, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area > largest_area:
            largest_area = area
            largest_label = label_id

    return (labels == largest_label).astype(np.uint8)


def remove_small_components(binary_mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1 or binary_mask.sum() == 0:
        return binary_mask.copy()

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    output = np.zeros_like(binary_mask)
    for label_id in range(1, num_labels):
        if stats[label_id, cv2.CC_STAT_AREA] >= min_area:
            output[labels == label_id] = 1
    return output


def clean_binary_mask(binary_mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    if binary_mask.sum() == 0 or kernel_size <= 1:
        return binary_mask.copy()

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    return cleaned


def compute_mask_centroid(binary_mask: np.ndarray) -> tuple[Optional[float], Optional[float]]:
    if binary_mask.sum() == 0:
        return None, None

    moments = cv2.moments(binary_mask)
    if abs(moments["m00"]) < 1e-8:
        return None, None

    return float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])


def fit_ellipse_to_mask(binary_mask: np.ndarray) -> Optional[EllipseResult]:
    if binary_mask.sum() == 0:
        return None

    contour_img = (binary_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(contour_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return None

    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5:
        return None

    ellipse = cv2.fitEllipse(contour)
    (center_x, center_y), (axis_a, axis_b), angle_deg = ellipse

    major_axis = float(max(axis_a, axis_b))
    minor_axis = float(min(axis_a, axis_b))
    fixed_angle = float(angle_deg)
    if axis_b > axis_a:
        fixed_angle = (fixed_angle + 90.0) % 180.0

    return EllipseResult(
        center_x=float(center_x),
        center_y=float(center_y),
        major_axis=major_axis,
        minor_axis=minor_axis,
        angle_deg=fixed_angle,
    )


def extract_region_geometry(binary_mask: np.ndarray) -> RegionGeometry:
    area = int(binary_mask.sum())
    if area == 0:
        return build_empty_region_geometry()

    center_x, center_y = compute_mask_centroid(binary_mask)
    ellipse = fit_ellipse_to_mask(binary_mask)
    return RegionGeometry(area=area, center_x=center_x, center_y=center_y, ellipse=ellipse)


def rotate_vector(dx: float, dy: float, angle_deg: float) -> tuple[float, float]:
    theta = math.radians(angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    return cos_t * dx - sin_t * dy, sin_t * dx + cos_t * dy


def compute_mask_bounds(binary_mask: np.ndarray) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if binary_mask.sum() == 0:
        return None, None, None, None

    ys, xs = np.nonzero(binary_mask)
    if xs.size == 0 or ys.size == 0:
        return None, None, None, None

    min_x = float(xs.min())
    max_x = float(xs.max())
    min_y = float(ys.min())
    max_y = float(ys.max())
    return min_x, min_y, max_x, max_y


def normalize_point_in_region(
    x: Optional[float],
    y: Optional[float],
    region_mask: np.ndarray,
    region_geometry: RegionGeometry,
) -> tuple[Optional[float], Optional[float], bool]:
    if x is None or y is None or region_geometry.area <= 0:
        return None, None, False

    if region_geometry.ellipse is not None:
        center_x = region_geometry.ellipse.center_x
        center_y = region_geometry.ellipse.center_y
        local_x, local_y = rotate_vector(
            float(x - center_x),
            float(y - center_y),
            angle_deg=-region_geometry.ellipse.angle_deg,
        )

        semi_major = region_geometry.ellipse.major_axis / 2.0
        semi_minor = region_geometry.ellipse.minor_axis / 2.0
        if semi_major > 1e-6 and semi_minor > 1e-6:
            return float(local_x / semi_major), float(local_y / semi_minor), True

    min_x, min_y, max_x, max_y = compute_mask_bounds(region_mask)
    if min_x is None or min_y is None or max_x is None or max_y is None:
        return None, None, False

    half_width = (max_x - min_x) / 2.0
    half_height = (max_y - min_y) / 2.0
    if half_width <= 1e-6 or half_height <= 1e-6:
        return None, None, False

    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    return float((x - center_x) / half_width), float((y - center_y) / half_height), True


def normalize_point_in_ellipse(
    x: Optional[float],
    y: Optional[float],
    ellipse: Optional[EllipseResult],
) -> tuple[Optional[float], Optional[float], bool]:
    if x is None or y is None or ellipse is None:
        return None, None, False

    local_x, local_y = rotate_vector(
        float(x - ellipse.center_x),
        float(y - ellipse.center_y),
        angle_deg=-ellipse.angle_deg,
    )
    semi_major = ellipse.major_axis / 2.0
    semi_minor = ellipse.minor_axis / 2.0
    if semi_major <= 1e-6 or semi_minor <= 1e-6:
        return None, None, False

    return float(local_x / semi_major), float(local_y / semi_minor), True


def extract_gaze_features_from_label_map(
    pred_label_map: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    outer_boundary_class_id: int = 1,
    iris_class_id: int = 2,
    pupil_class_id: int = 3,
    kernel_size: int = 3,
    iris_min_area: int = 100,
    pupil_min_area: int = 20,
    max_valid_norm_radius: float = 0.85,
    outer_min_extra_area: int = 24,
) -> GazeFeatureResult:
    masked_label_map = apply_valid_mask(pred_label_map, valid_mask)

    iris_mask = extract_class_binary_mask(masked_label_map, iris_class_id)
    pupil_mask = extract_class_binary_mask(masked_label_map, pupil_class_id)
    boundary_mask = extract_class_binary_mask(masked_label_map, outer_boundary_class_id)
    boundary_plus_iris_mask = np.logical_or(boundary_mask > 0, iris_mask > 0).astype(np.uint8)
    outer_eye_mask = extract_non_background_mask(masked_label_map)

    iris_mask = remove_small_components(iris_mask, iris_min_area)
    pupil_mask = remove_small_components(pupil_mask, pupil_min_area)
    boundary_mask = remove_small_components(boundary_mask, min_area=outer_min_extra_area)
    boundary_plus_iris_mask = remove_small_components(boundary_plus_iris_mask, min_area=max(iris_min_area, outer_min_extra_area))
    outer_eye_mask = remove_small_components(outer_eye_mask, min_area=max(iris_min_area, outer_min_extra_area))

    iris_mask = keep_largest_connected_component(iris_mask)
    pupil_mask = keep_largest_connected_component(pupil_mask)
    boundary_mask = keep_largest_connected_component(boundary_mask)
    boundary_plus_iris_mask = keep_largest_connected_component(boundary_plus_iris_mask)
    outer_eye_mask = keep_largest_connected_component(outer_eye_mask)

    iris_mask = clean_binary_mask(iris_mask, kernel_size=kernel_size)
    pupil_mask = clean_binary_mask(pupil_mask, kernel_size=kernel_size)
    boundary_mask = clean_binary_mask(boundary_mask, kernel_size=kernel_size)
    boundary_plus_iris_mask = clean_binary_mask(boundary_plus_iris_mask, kernel_size=kernel_size)
    outer_eye_mask = clean_binary_mask(outer_eye_mask, kernel_size=kernel_size)

    iris_geometry = extract_region_geometry(iris_mask)
    pupil_geometry = extract_region_geometry(pupil_mask)
    boundary_geometry = extract_region_geometry(boundary_mask)
    boundary_plus_iris_geometry = extract_region_geometry(boundary_plus_iris_mask)
    outer_eye_geometry = extract_region_geometry(outer_eye_mask)

    offset_dx = None
    offset_dy = None
    local_dx = None
    local_dy = None
    norm_dx = None
    norm_dy = None
    norm_radius = None
    iris_feature_x = None
    iris_feature_y = None
    iris_outer_feature_x = None
    iris_outer_feature_y = None
    iris_outer_valid = False
    iris_only_valid = False
    area_ratio = None
    invalid_reasons: list[str] = []

    if iris_geometry.area <= 0:
        invalid_reasons.append("iris_mask_empty")
    if pupil_geometry.area <= 0:
        invalid_reasons.append("pupil_mask_empty")
    if iris_geometry.center_x is None or iris_geometry.center_y is None:
        invalid_reasons.append("iris_center_missing")
    if pupil_geometry.center_x is None or pupil_geometry.center_y is None:
        invalid_reasons.append("pupil_center_missing")

    if (
        iris_geometry.center_x is not None
        and iris_geometry.center_y is not None
        and pupil_geometry.center_x is not None
        and pupil_geometry.center_y is not None
    ):
        offset_dx = float(pupil_geometry.center_x - iris_geometry.center_x)
        offset_dy = float(pupil_geometry.center_y - iris_geometry.center_y)

    if iris_geometry.ellipse is not None and offset_dx is not None and offset_dy is not None:
        local_dx, local_dy = rotate_vector(offset_dx, offset_dy, angle_deg=-iris_geometry.ellipse.angle_deg)

        semi_major = iris_geometry.ellipse.major_axis / 2.0
        semi_minor = iris_geometry.ellipse.minor_axis / 2.0

        if semi_major > 1e-6:
            norm_dx = float(local_dx / semi_major)
        if semi_minor > 1e-6:
            norm_dy = float(local_dy / semi_minor)
        if norm_dx is not None and norm_dy is not None:
            norm_radius = float(math.sqrt(norm_dx * norm_dx + norm_dy * norm_dy))

    if iris_geometry.area > 0:
        area_ratio = float(pupil_geometry.area / iris_geometry.area)

    iris_feature_center_x = iris_geometry.ellipse.center_x if iris_geometry.ellipse is not None else iris_geometry.center_x
    iris_feature_center_y = iris_geometry.ellipse.center_y if iris_geometry.ellipse is not None else iris_geometry.center_y
    height, width = masked_label_map.shape[:2]
    if iris_feature_center_x is not None and iris_feature_center_y is not None and width > 1 and height > 1:
        iris_feature_x = float((iris_feature_center_x / float(width - 1)) * 2.0 - 1.0)
        iris_feature_y = float((iris_feature_center_y / float(height - 1)) * 2.0 - 1.0)
        iris_only_valid = True

    reference_mask = boundary_plus_iris_mask
    reference_geometry = boundary_plus_iris_geometry
    reference_area = int(reference_mask.sum())

    if reference_area < outer_min_extra_area:
        reference_mask = boundary_mask
        reference_geometry = boundary_geometry
        reference_area = int(reference_mask.sum())

    if reference_area < outer_min_extra_area:
        outer_support_mask = outer_eye_mask.copy()
        outer_support_mask[iris_mask > 0] = 0
        outer_support_mask[pupil_mask > 0] = 0
        outer_support_area = int(outer_support_mask.sum())
        if outer_support_area >= outer_min_extra_area:
            reference_mask = outer_support_mask
            reference_geometry = extract_region_geometry(reference_mask)
            reference_area = int(reference_mask.sum())

    if reference_area >= outer_min_extra_area:
        iris_outer_feature_x, iris_outer_feature_y, iris_outer_valid = normalize_point_in_region(
            x=iris_feature_center_x,
            y=iris_feature_center_y,
            region_mask=reference_mask,
            region_geometry=reference_geometry,
        )

    if iris_geometry.ellipse is None:
        invalid_reasons.append("iris_no_ellipse")
    if pupil_geometry.ellipse is None:
        invalid_reasons.append("pupil_no_ellipse")
    if offset_dx is None or offset_dy is None:
        invalid_reasons.append("offset_missing")
    if norm_dx is None:
        invalid_reasons.append("norm_dx_missing")
    if norm_dy is None:
        invalid_reasons.append("norm_dy_missing")
    if norm_radius is None:
        invalid_reasons.append("norm_radius_missing")
    elif norm_radius > max_valid_norm_radius:
        invalid_reasons.append(f"norm_radius_exceeds:{norm_radius:.3f}>{max_valid_norm_radius:.3f}")

    valid = (
        iris_geometry.ellipse is not None
        and pupil_geometry.ellipse is not None
        and norm_dx is not None
        and norm_dy is not None
        and norm_radius is not None
        and norm_radius <= max_valid_norm_radius
    )

    return GazeFeatureResult(
        iris_mask=iris_mask,
        pupil_mask=pupil_mask,
        iris_geometry=iris_geometry,
        pupil_geometry=pupil_geometry,
        outer_boundary_geometry=boundary_plus_iris_geometry,
        offset_dx=offset_dx,
        offset_dy=offset_dy,
        local_dx=local_dx,
        local_dy=local_dy,
        norm_dx=norm_dx,
        norm_dy=norm_dy,
        norm_radius=norm_radius,
        iris_feature_x=iris_feature_x,
        iris_feature_y=iris_feature_y,
        iris_outer_feature_x=iris_outer_feature_x,
        iris_outer_feature_y=iris_outer_feature_y,
        iris_outer_valid=iris_outer_valid,
        iris_only_valid=iris_only_valid,
        pupil_iris_area_ratio=area_ratio,
        invalid_reasons=tuple(invalid_reasons),
        valid=valid,
    )


def resolve_tracking_features(
    result: GazeFeatureResult,
    feature_mode: str,
    reference_boundary_ellipse: Optional[EllipseResult] = None,
) -> tuple[Optional[float], Optional[float], bool, tuple[str, ...]]:
    normalized_mode = feature_mode.strip().lower()

    if normalized_mode == "pupil_iris":
        reasons = tuple() if result.valid else result.invalid_reasons
        return result.norm_dx, result.norm_dy, result.valid, reasons

    if normalized_mode == "iris_only":
        iris_center_x = result.iris_geometry.ellipse.center_x if result.iris_geometry.ellipse is not None else result.iris_geometry.center_x
        iris_center_y = result.iris_geometry.ellipse.center_y if result.iris_geometry.ellipse is not None else result.iris_geometry.center_y

        ref_feature_x, ref_feature_y, ref_valid = normalize_point_in_ellipse(
            x=iris_center_x,
            y=iris_center_y,
            ellipse=reference_boundary_ellipse,
        )
        if ref_valid and ref_feature_x is not None and ref_feature_y is not None:
            return ref_feature_x, ref_feature_y, True, tuple()

        if (
            result.iris_outer_valid
            and result.iris_outer_feature_x is not None
            and result.iris_outer_feature_y is not None
        ):
            return result.iris_outer_feature_x, result.iris_outer_feature_y, True, tuple()

        reasons: list[str] = []
        if result.iris_feature_x is None:
            reasons.append("iris_feature_x_missing")
        if result.iris_feature_y is None:
            reasons.append("iris_feature_y_missing")
        valid = bool(result.iris_only_valid and result.iris_feature_x is not None and result.iris_feature_y is not None)
        return result.iris_feature_x, result.iris_feature_y, valid, tuple(reasons)

    raise ValueError(f"不支持的特征模式: {feature_mode}")


def build_five_point_calibration_points(margin: float = 0.1) -> list[CalibrationPoint]:
    margin = float(np.clip(margin, 0.0, 0.45))
    return [
        CalibrationPoint(name="top_left", u=margin, v=margin),
        CalibrationPoint(name="top_right", u=1.0 - margin, v=margin),
        CalibrationPoint(name="bottom_right", u=1.0 - margin, v=1.0 - margin),
        CalibrationPoint(name="bottom_left", u=margin, v=1.0 - margin),
        CalibrationPoint(name="center", u=0.5, v=0.5),
    ]


def build_nine_point_calibration_points(margin: float = 0.1) -> list[CalibrationPoint]:
    margin = float(np.clip(margin, 0.0, 0.45))
    left = margin
    right = 1.0 - margin
    top = margin
    bottom = 1.0 - margin
    center = 0.5

    return [
        CalibrationPoint(name="top_left", u=left, v=top),
        CalibrationPoint(name="top_center", u=center, v=top),
        CalibrationPoint(name="top_right", u=right, v=top),
        CalibrationPoint(name="middle_right", u=right, v=center),
        CalibrationPoint(name="bottom_right", u=right, v=bottom),
        CalibrationPoint(name="bottom_center", u=center, v=bottom),
        CalibrationPoint(name="bottom_left", u=left, v=bottom),
        CalibrationPoint(name="middle_left", u=left, v=center),
        CalibrationPoint(name="center", u=center, v=center),
    ]


def begin_calibration_session(
    now_ms: float,
    settle_ms: int,
    capture_ms: int,
    min_valid_frames: int,
    margin: float = 0.1,
    point_pattern: str = "five",
) -> CalibrationSession:
    normalized_pattern = point_pattern.strip().lower()
    if normalized_pattern == "five":
        points = build_five_point_calibration_points(margin=margin)
    elif normalized_pattern == "nine":
        points = build_nine_point_calibration_points(margin=margin)
    else:
        raise ValueError(f"不支持的校准点布局: {point_pattern}")

    return CalibrationSession(
        points=points,
        settle_ms=int(settle_ms),
        capture_ms=int(capture_ms),
        min_valid_frames=int(min_valid_frames),
        state="settling",
        point_index=0,
        phase_started_at_ms=float(now_ms),
    )


def cancel_calibration_session(session: CalibrationSession, reason: str) -> CalibrationSession:
    session.state = "canceled"
    session.failure_reason = reason
    session.current_point_samples.clear()
    session.boundary_samples.clear()
    session.collected_samples.clear()
    session.model = None
    session.phase_started_at_ms = None
    return session


def fail_calibration_session(session: CalibrationSession, reason: str) -> CalibrationSession:
    session.state = "failed"
    session.failure_reason = reason
    session.current_point_samples.clear()
    session.boundary_samples.clear()
    session.model = None
    session.phase_started_at_ms = None
    return session


def build_static_boundary_ellipse(
    boundary_samples: Sequence[tuple[float, float, float, float, float]],
) -> Optional[EllipseResult]:
    if len(boundary_samples) == 0:
        return None

    samples = np.array(boundary_samples, dtype=np.float64)
    valid_axis_mask = (samples[:, 2] > 1e-6) & (samples[:, 3] > 1e-6)
    samples = samples[valid_axis_mask]
    if samples.shape[0] == 0:
        return None

    # 先按中心点稳定性去掉离群帧，减少眨眼/遮挡对静态眼眶的影响。
    center_xy = samples[:, :2]
    center_xy_median = np.median(center_xy, axis=0)
    center_distance = np.linalg.norm(center_xy - center_xy_median[np.newaxis, :], axis=1)
    if samples.shape[0] >= 5:
        keep_mask = center_distance <= np.percentile(center_distance, 85.0)
    else:
        keep_mask = np.ones(samples.shape[0], dtype=bool)
    if np.any(keep_mask):
        samples = samples[keep_mask]

    major_axis_values = samples[:, 2]
    minor_axis_values = samples[:, 3]
    if samples.shape[0] >= 5:
        major_low, major_high = np.percentile(major_axis_values, [10.0, 95.0])
        minor_low, minor_high = np.percentile(minor_axis_values, [10.0, 95.0])
        axis_keep_mask = (
            (major_axis_values >= major_low)
            & (major_axis_values <= major_high)
            & (minor_axis_values >= minor_low)
            & (minor_axis_values <= minor_high)
        )
        if np.any(axis_keep_mask):
            samples = samples[axis_keep_mask]
            major_axis_values = samples[:, 2]
            minor_axis_values = samples[:, 3]

    center_x = float(np.median(samples[:, 0]))
    center_y = float(np.median(samples[:, 1]))
    major_axis = float(np.quantile(major_axis_values, 0.70))
    minor_axis = float(np.quantile(minor_axis_values, 0.70))
    if minor_axis > major_axis:
        major_axis, minor_axis = minor_axis, major_axis
    if major_axis <= 1e-6 or minor_axis <= 1e-6:
        return None

    doubled_angles = np.deg2rad(samples[:, 4] * 2.0)
    mean_sin = float(np.mean(np.sin(doubled_angles)))
    mean_cos = float(np.mean(np.cos(doubled_angles)))
    angle_deg = float((0.5 * np.degrees(np.arctan2(mean_sin, mean_cos))) % 180.0)

    return EllipseResult(
        center_x=center_x,
        center_y=center_y,
        major_axis=major_axis,
        minor_axis=minor_axis,
        angle_deg=angle_deg,
    )


def fit_five_point_affine(
    samples: Sequence[CalibrationSample],
    expected_point_count: int = 5,
) -> CalibrationModel:
    if len(samples) != expected_point_count:
        raise ValueError(f"校准点数量不足，期望 {expected_point_count} 个，实际 {len(samples)} 个。")

    point_names = [sample.point_name for sample in samples]
    if len(set(point_names)) != expected_point_count:
        raise ValueError("校准点不完整或存在重复，无法拟合仿射映射。")

    source = np.array(
        [[sample.feature_dx, sample.feature_dy, 1.0] for sample in samples],
        dtype=np.float64,
    )
    target = np.array(
        [[sample.target_u, sample.target_v] for sample in samples],
        dtype=np.float64,
    )
    weights = np.array([float(np.clip(sample.weight, 0.05, 1.0)) for sample in samples], dtype=np.float64)

    if np.linalg.matrix_rank(source) < 3:
        raise ValueError("校准样本矩阵退化，无法拟合仿射映射。")

    weighted_source = source * np.sqrt(weights)[:, np.newaxis]
    weighted_target = target * np.sqrt(weights)[:, np.newaxis]
    solution, _, _, _ = np.linalg.lstsq(weighted_source, weighted_target, rcond=None)
    return CalibrationModel(
        matrix=solution.T,
        point_names=tuple(point_names),
        source_features=source[:, :2],
        target_points=target,
        mapping_type="affine",
    )


def fit_nine_point_polynomial(
    samples: Sequence[CalibrationSample],
    expected_point_count: int = 9,
    ridge_lambda: float = 1e-4,
) -> CalibrationModel:
    if len(samples) != expected_point_count:
        raise ValueError(f"校准点数量不足，期望 {expected_point_count} 个，实际 {len(samples)} 个。")

    point_names = [sample.point_name for sample in samples]
    if len(set(point_names)) != expected_point_count:
        raise ValueError("校准点不完整或存在重复，无法拟合二次映射。")

    source = np.array(
        [
            [
                sample.feature_dx,
                sample.feature_dy,
                sample.feature_dx * sample.feature_dx,
                sample.feature_dx * sample.feature_dy,
                sample.feature_dy * sample.feature_dy,
                1.0,
            ]
            for sample in samples
        ],
        dtype=np.float64,
    )
    target = np.array(
        [[sample.target_u, sample.target_v] for sample in samples],
        dtype=np.float64,
    )
    weights = np.array([float(np.clip(sample.weight, 0.05, 1.0)) for sample in samples], dtype=np.float64)

    if np.linalg.matrix_rank(source) < 6:
        raise ValueError("校准样本矩阵退化，无法拟合二次映射。")

    weighted_source = source * np.sqrt(weights)[:, np.newaxis]
    weighted_target = target * np.sqrt(weights)[:, np.newaxis]
    if ridge_lambda > 0.0:
        reg = np.eye(source.shape[1], dtype=np.float64)
        reg[-1, -1] = 0.0
        weighted_source = np.vstack([weighted_source, np.sqrt(ridge_lambda) * reg])
        weighted_target = np.vstack([weighted_target, np.zeros((source.shape[1], target.shape[1]), dtype=np.float64)])

    solution, _, _, _ = np.linalg.lstsq(weighted_source, weighted_target, rcond=None)
    return CalibrationModel(
        matrix=solution.T,
        point_names=tuple(point_names),
        source_features=source[:, :2],
        target_points=target,
        mapping_type="poly2",
    )


def fit_calibration_mapping(
    samples: Sequence[CalibrationSample],
    expected_point_count: int,
    prefer_polynomial: bool = True,
    ridge_lambda: float = 1e-4,
) -> CalibrationModel:
    if prefer_polynomial and expected_point_count >= 6:
        try:
            return fit_nine_point_polynomial(
                samples=samples,
                expected_point_count=expected_point_count,
                ridge_lambda=ridge_lambda,
            )
        except ValueError:
            # 当二次模型退化时回退到仿射，保证可用性。
            pass

    return fit_five_point_affine(samples=samples, expected_point_count=expected_point_count)


def _build_model_feature_vector(model: CalibrationModel, feature_dx: float, feature_dy: float) -> np.ndarray:
    if model.mapping_type == "poly2":
        return np.array(
            [
                feature_dx,
                feature_dy,
                feature_dx * feature_dx,
                feature_dx * feature_dy,
                feature_dy * feature_dy,
                1.0,
            ],
            dtype=np.float64,
        )

    return np.array([feature_dx, feature_dy, 1.0], dtype=np.float64)


def predict_screen_point(
    model: Optional[CalibrationModel],
    feature_dx: Optional[float],
    feature_dy: Optional[float],
    clamp: bool = True,
) -> Optional[tuple[float, float]]:
    if model is None or feature_dx is None or feature_dy is None:
        return None

    feature_vector = _build_model_feature_vector(model, float(feature_dx), float(feature_dy))
    uv = model.matrix @ feature_vector
    u = float(uv[0])
    v = float(uv[1])

    if clamp:
        u = float(np.clip(u, 0.0, 1.0))
        v = float(np.clip(v, 0.0, 1.0))

    return u, v


def advance_calibration_session(
    session: CalibrationSession,
    now_ms: float,
    feature_dx: Optional[float],
    feature_dy: Optional[float],
    feature_valid: bool,
    feature_confidence: Optional[float] = None,
    boundary_ellipse: Optional[EllipseResult] = None,
) -> CalibrationSession:
    if not session.is_active:
        return session

    if session.phase_started_at_ms is None:
        session.phase_started_at_ms = float(now_ms)

    elapsed_ms = float(now_ms) - float(session.phase_started_at_ms)

    if session.state == "settling":
        if elapsed_ms >= session.settle_ms:
            session.state = "capturing"
            session.phase_started_at_ms = float(now_ms)
            session.current_point_samples.clear()
        return session

    if session.state != "capturing":
        return session

    if boundary_ellipse is not None:
        if boundary_ellipse.major_axis > 1e-6 and boundary_ellipse.minor_axis > 1e-6:
            session.boundary_samples.append(
                (
                    float(boundary_ellipse.center_x),
                    float(boundary_ellipse.center_y),
                    float(boundary_ellipse.major_axis),
                    float(boundary_ellipse.minor_axis),
                    float(boundary_ellipse.angle_deg),
                )
            )

    if feature_valid and feature_dx is not None and feature_dy is not None:
        confidence = 1.0 if feature_confidence is None else float(np.clip(feature_confidence, 0.0, 1.0))
        session.current_point_samples.append((float(feature_dx), float(feature_dy), confidence))

    if elapsed_ms < session.capture_ms:
        return session

    point = session.current_point
    if point is None:
        return fail_calibration_session(session, "校准点索引异常，无法继续。")

    if len(session.current_point_samples) < session.min_valid_frames:
        return fail_calibration_session(
            session,
            f"{point.name} 采样失败：有效帧不足 {session.min_valid_frames}。",
        )

    point_samples = np.array(session.current_point_samples, dtype=np.float64)
    point_xy = point_samples[:, :2]
    point_weights = np.clip(point_samples[:, 2], 0.05, 1.0)

    robust_center = np.median(point_xy, axis=0)
    distance = np.linalg.norm(point_xy - robust_center[np.newaxis, :], axis=1)
    if point_xy.shape[0] >= 5:
        keep_mask = distance <= np.percentile(distance, 80.0)
    else:
        keep_mask = np.ones(point_xy.shape[0], dtype=bool)
    if not np.any(keep_mask):
        keep_mask = np.ones(point_xy.shape[0], dtype=bool)

    filtered_xy = point_xy[keep_mask]
    filtered_weights = point_weights[keep_mask]
    mean_dx = float(np.average(filtered_xy[:, 0], weights=filtered_weights))
    mean_dy = float(np.average(filtered_xy[:, 1], weights=filtered_weights))
    sample_weight = float(np.clip(np.mean(filtered_weights), 0.05, 1.0))

    session.collected_samples.append(
        CalibrationSample(
            point_name=point.name,
            target_u=point.u,
            target_v=point.v,
            feature_dx=mean_dx,
            feature_dy=mean_dy,
            weight=sample_weight,
        )
    )

    session.point_index += 1
    session.current_point_samples.clear()

    if session.point_index >= len(session.points):
        try:
            session.model = fit_calibration_mapping(
                session.collected_samples,
                expected_point_count=len(session.points),
                prefer_polynomial=True,
            )
            session.model.static_boundary_ellipse = build_static_boundary_ellipse(session.boundary_samples)
        except ValueError as exc:
            return fail_calibration_session(session, str(exc))

        session.state = "completed"
        session.failure_reason = None
        session.phase_started_at_ms = None
        return session

    session.state = "settling"
    session.phase_started_at_ms = float(now_ms)
    return session
