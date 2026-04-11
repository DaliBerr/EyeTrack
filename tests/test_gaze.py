import unittest

import cv2
import numpy as np

from eyetrack.gaze import (
    CalibrationSample,
    advance_calibration_session,
    begin_calibration_session,
    build_five_point_calibration_points,
    extract_gaze_features_from_label_map,
    fit_five_point_affine,
    predict_screen_point,
    resolve_tracking_features,
)


class GazeCalibrationTests(unittest.TestCase):
    def test_fit_five_point_affine_recovers_known_mapping(self) -> None:
        points = build_five_point_calibration_points(margin=0.1)
        linear = np.array([[0.8, -0.1], [0.15, 0.7]], dtype=np.float64)
        bias = np.array([0.2, 0.15], dtype=np.float64)

        samples = []
        for point in points:
            target = np.array([point.u, point.v], dtype=np.float64)
            feature = np.linalg.solve(linear, target - bias)
            samples.append(
                CalibrationSample(
                    point_name=point.name,
                    target_u=point.u,
                    target_v=point.v,
                    feature_dx=float(feature[0]),
                    feature_dy=float(feature[1]),
                )
            )

        model = fit_five_point_affine(samples)
        feature_test = np.array([0.35, -0.2], dtype=np.float64)
        expected = linear @ feature_test + bias
        predicted = predict_screen_point(model, float(feature_test[0]), float(feature_test[1]), clamp=False)

        self.assertIsNotNone(predicted)
        self.assertAlmostEqual(predicted[0], expected[0], places=6)
        self.assertAlmostEqual(predicted[1], expected[1], places=6)

    def test_fit_five_point_affine_requires_all_points(self) -> None:
        points = build_five_point_calibration_points(margin=0.1)
        samples = [
            CalibrationSample(
                point_name=point.name,
                target_u=point.u,
                target_v=point.v,
                feature_dx=float(index),
                feature_dy=float(index + 1),
            )
            for index, point in enumerate(points[:4])
        ]

        with self.assertRaisesRegex(ValueError, "校准点数量不足"):
            fit_five_point_affine(samples)

    def test_fit_five_point_affine_rejects_degenerate_source(self) -> None:
        points = build_five_point_calibration_points(margin=0.1)
        samples = [
            CalibrationSample(
                point_name=point.name,
                target_u=point.u,
                target_v=point.v,
                feature_dx=float(index),
                feature_dy=float(index * 2),
            )
            for index, point in enumerate(points)
        ]

        with self.assertRaisesRegex(ValueError, "退化"):
            fit_five_point_affine(samples)

    def test_predict_screen_point_returns_none_without_model(self) -> None:
        self.assertIsNone(predict_screen_point(None, 0.1, -0.2))

    def test_extract_gaze_features_reports_invalid_reasons_for_empty_mask(self) -> None:
        label_map = np.zeros((32, 32), dtype=np.uint8)
        result = extract_gaze_features_from_label_map(
            pred_label_map=label_map,
            iris_min_area=1,
            pupil_min_area=1,
        )

        self.assertFalse(result.valid)
        self.assertIsNone(result.norm_dx)
        self.assertIn("iris_mask_empty", result.invalid_reasons)
        self.assertIn("pupil_mask_empty", result.invalid_reasons)
        self.assertIn("iris_no_ellipse", result.invalid_reasons)
        self.assertIn("pupil_no_ellipse", result.invalid_reasons)

    def test_resolve_tracking_features_supports_iris_only(self) -> None:
        label_map = np.zeros((48, 64), dtype=np.uint8)
        cv2.ellipse(label_map, center=(32, 20), axes=(12, 8), angle=0, startAngle=0, endAngle=360, color=2, thickness=-1)
        result = extract_gaze_features_from_label_map(
            pred_label_map=label_map,
            iris_min_area=1,
            pupil_min_area=1,
        )

        feature_x, feature_y, feature_valid, feature_reasons = resolve_tracking_features(result, "iris_only")

        self.assertTrue(feature_valid)
        self.assertEqual(feature_reasons, tuple())
        self.assertIsNotNone(feature_x)
        self.assertIsNotNone(feature_y)

        norm_x, norm_y, norm_valid, norm_reasons = resolve_tracking_features(result, "pupil_iris")
        self.assertIsNone(norm_x)
        self.assertIsNone(norm_y)
        self.assertFalse(norm_valid)
        self.assertIn("pupil_mask_empty", norm_reasons)

    def test_calibration_session_fails_on_insufficient_valid_frames(self) -> None:
        session = begin_calibration_session(
            now_ms=0.0,
            settle_ms=0,
            capture_ms=10,
            min_valid_frames=3,
            margin=0.1,
        )

        session = advance_calibration_session(session, now_ms=0.0, feature_dx=None, feature_dy=None, feature_valid=False)
        session = advance_calibration_session(session, now_ms=1.0, feature_dx=0.1, feature_dy=0.2, feature_valid=True)
        session = advance_calibration_session(session, now_ms=11.0, feature_dx=None, feature_dy=None, feature_valid=False)

        self.assertEqual(session.state, "failed")
        self.assertIsNotNone(session.failure_reason)
        self.assertIn("有效帧不足", session.failure_reason)

    def test_calibration_session_completes_and_predicts(self) -> None:
        linear = np.array([[0.8, -0.1], [0.15, 0.7]], dtype=np.float64)
        bias = np.array([0.2, 0.15], dtype=np.float64)

        session = begin_calibration_session(
            now_ms=0.0,
            settle_ms=0,
            capture_ms=10,
            min_valid_frames=2,
            margin=0.1,
        )

        current_time = 0.0
        for point in session.points:
            target = np.array([point.u, point.v], dtype=np.float64)
            feature = np.linalg.solve(linear, target - bias)

            session = advance_calibration_session(session, now_ms=current_time, feature_dx=None, feature_dy=None, feature_valid=False)
            session = advance_calibration_session(session, now_ms=current_time + 1.0, feature_dx=float(feature[0]), feature_dy=float(feature[1]), feature_valid=True)
            session = advance_calibration_session(session, now_ms=current_time + 2.0, feature_dx=float(feature[0]), feature_dy=float(feature[1]), feature_valid=True)
            session = advance_calibration_session(session, now_ms=current_time + 11.0, feature_dx=None, feature_dy=None, feature_valid=False)
            current_time += 20.0

        self.assertEqual(session.state, "completed")
        self.assertIsNotNone(session.model)

        feature_test = np.array([0.05, 0.1], dtype=np.float64)
        expected = linear @ feature_test + bias
        predicted = predict_screen_point(session.model, float(feature_test[0]), float(feature_test[1]), clamp=False)

        self.assertIsNotNone(predicted)
        self.assertAlmostEqual(predicted[0], expected[0], places=6)
        self.assertAlmostEqual(predicted[1], expected[1], places=6)


if __name__ == "__main__":
    unittest.main()
