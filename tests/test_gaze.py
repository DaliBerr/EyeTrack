import unittest

import cv2
import numpy as np

from eyetrack.gaze import (
    CalibrationSample,
    EllipseResult,
    advance_calibration_session,
    begin_calibration_session,
    build_five_point_calibration_points,
    build_nine_point_calibration_points,
    extract_gaze_features_from_label_map,
    fit_calibration_mapping,
    fit_five_point_affine,
    predict_screen_point,
    resolve_tracking_features,
)


class GazeCalibrationTests(unittest.TestCase):
    def test_fit_calibration_mapping_uses_poly2_for_nine_points(self) -> None:
        polynomial_coeff_u = np.array([0.52, 0.31, -0.11, 0.08, 0.05, -0.04], dtype=np.float64)
        polynomial_coeff_v = np.array([0.47, 0.09, 0.28, -0.06, 0.03, 0.07], dtype=np.float64)
        feature_grid = [
            (-0.8, -0.8),
            (0.0, -0.8),
            (0.8, -0.8),
            (-0.8, 0.0),
            (0.0, 0.0),
            (0.8, 0.0),
            (-0.8, 0.8),
            (0.0, 0.8),
            (0.8, 0.8),
        ]

        samples = []
        for index, (feature_dx, feature_dy) in enumerate(feature_grid):
            terms = np.array(
                [
                    1.0,
                    feature_dx,
                    feature_dy,
                    feature_dx * feature_dx,
                    feature_dx * feature_dy,
                    feature_dy * feature_dy,
                ],
                dtype=np.float64,
            )
            samples.append(
                CalibrationSample(
                    point_name=f"p{index}",
                    target_u=float(polynomial_coeff_u @ terms),
                    target_v=float(polynomial_coeff_v @ terms),
                    feature_dx=float(feature_dx),
                    feature_dy=float(feature_dy),
                )
            )

        model = fit_calibration_mapping(samples, expected_point_count=9, prefer_polynomial=True)
        self.assertEqual(model.mapping_type, "poly2")

        feature_test = np.array([0.33, -0.27], dtype=np.float64)
        test_terms = np.array(
            [
                1.0,
                feature_test[0],
                feature_test[1],
                feature_test[0] * feature_test[0],
                feature_test[0] * feature_test[1],
                feature_test[1] * feature_test[1],
            ],
            dtype=np.float64,
        )
        expected_u = float(polynomial_coeff_u @ test_terms)
        expected_v = float(polynomial_coeff_v @ test_terms)
        predicted = predict_screen_point(model, float(feature_test[0]), float(feature_test[1]), clamp=False)

        self.assertIsNotNone(predicted)
        self.assertAlmostEqual(predicted[0], expected_u, places=5)
        self.assertAlmostEqual(predicted[1], expected_v, places=5)

    def test_build_nine_point_calibration_points_includes_edge_centers(self) -> None:
        points = build_nine_point_calibration_points(margin=0.1)
        self.assertEqual(len(points), 9)

        point_map = {point.name: (point.u, point.v) for point in points}
        self.assertIn("top_center", point_map)
        self.assertIn("middle_right", point_map)
        self.assertIn("bottom_center", point_map)
        self.assertIn("middle_left", point_map)

        self.assertAlmostEqual(point_map["top_center"][0], 0.5, places=6)
        self.assertAlmostEqual(point_map["top_center"][1], 0.1, places=6)
        self.assertAlmostEqual(point_map["middle_right"][0], 0.9, places=6)
        self.assertAlmostEqual(point_map["middle_right"][1], 0.5, places=6)
        self.assertAlmostEqual(point_map["bottom_center"][0], 0.5, places=6)
        self.assertAlmostEqual(point_map["bottom_center"][1], 0.9, places=6)
        self.assertAlmostEqual(point_map["middle_left"][0], 0.1, places=6)
        self.assertAlmostEqual(point_map["middle_left"][1], 0.5, places=6)

    def test_begin_calibration_session_supports_nine_point_pattern(self) -> None:
        session = begin_calibration_session(
            now_ms=0.0,
            settle_ms=300,
            capture_ms=600,
            min_valid_frames=5,
            margin=0.1,
            point_pattern="nine",
        )

        self.assertEqual(session.state, "settling")
        self.assertEqual(len(session.points), 9)
        self.assertEqual(session.calibration_step, "1/9 top_left")

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

        with self.assertRaisesRegex(ValueError, "Insufficient calibration points"):
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

        with self.assertRaisesRegex(ValueError, " "):
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

    def test_outer_boundary_geometry_uses_boundary_plus_iris_union(self) -> None:
        label_map = np.zeros((80, 120), dtype=np.uint8)
        cv2.ellipse(label_map, center=(60, 40), axes=(34, 14), angle=0, startAngle=0, endAngle=360, color=1, thickness=-1)
        cv2.ellipse(label_map, center=(78, 34), axes=(12, 10), angle=0, startAngle=0, endAngle=360, color=2, thickness=-1)

        result = extract_gaze_features_from_label_map(
            pred_label_map=label_map,
            iris_min_area=1,
            pupil_min_area=1,
        )

        class1_area = int((label_map == 1).sum())
        union_area = int(np.logical_or(label_map == 1, label_map == 2).sum())
        self.assertGreater(union_area, class1_area)
        self.assertGreater(result.outer_boundary_geometry.area, class1_area)
        self.assertAlmostEqual(result.outer_boundary_geometry.area, union_area, delta=120)

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
        self.assertFalse(result.iris_outer_valid)
        self.assertEqual(feature_x, result.iris_feature_x)
        self.assertEqual(feature_y, result.iris_feature_y)

        norm_x, norm_y, norm_valid, norm_reasons = resolve_tracking_features(result, "pupil_iris")
        self.assertIsNone(norm_x)
        self.assertIsNone(norm_y)
        self.assertFalse(norm_valid)
        self.assertIn("pupil_mask_empty", norm_reasons)

    def test_resolve_tracking_features_prefers_outer_eye_reference_in_iris_only(self) -> None:
        label_map = np.zeros((64, 96), dtype=np.uint8)
        cv2.ellipse(label_map, center=(48, 32), axes=(30, 12), angle=0, startAngle=0, endAngle=360, color=1, thickness=-1)
        cv2.ellipse(label_map, center=(58, 27), axes=(10, 8), angle=0, startAngle=0, endAngle=360, color=2, thickness=-1)

        result = extract_gaze_features_from_label_map(
            pred_label_map=label_map,
            iris_min_area=1,
            pupil_min_area=1,
        )

        feature_x, feature_y, feature_valid, feature_reasons = resolve_tracking_features(result, "iris_only")

        self.assertTrue(feature_valid)
        self.assertEqual(feature_reasons, tuple())
        self.assertTrue(result.iris_outer_valid)
        self.assertIsNotNone(result.iris_outer_feature_x)
        self.assertIsNotNone(result.iris_outer_feature_y)
        self.assertAlmostEqual(feature_x, result.iris_outer_feature_x, places=6)
        self.assertAlmostEqual(feature_y, result.iris_outer_feature_y, places=6)
        self.assertGreater(feature_x, 0.0)
        self.assertLess(feature_y, 0.0)
        self.assertNotAlmostEqual(feature_x, result.iris_feature_x, places=2)

    def test_resolve_tracking_features_prefers_static_boundary_reference_when_available(self) -> None:
        label_map = np.zeros((64, 96), dtype=np.uint8)
        cv2.ellipse(label_map, center=(56, 26), axes=(10, 8), angle=0, startAngle=0, endAngle=360, color=2, thickness=-1)

        result = extract_gaze_features_from_label_map(
            pred_label_map=label_map,
            iris_min_area=1,
            pupil_min_area=1,
        )

        reference = EllipseResult(
            center_x=48.0,
            center_y=32.0,
            major_axis=40.0,
            minor_axis=24.0,
            angle_deg=0.0,
        )

        feature_x, feature_y, feature_valid, feature_reasons = resolve_tracking_features(
            result,
            "iris_only",
            reference_boundary_ellipse=reference,
        )

        self.assertTrue(feature_valid)
        self.assertEqual(feature_reasons, tuple())
        self.assertIsNotNone(feature_x)
        self.assertIsNotNone(feature_y)

        iris_center_x = result.iris_geometry.ellipse.center_x if result.iris_geometry.ellipse is not None else result.iris_geometry.center_x
        iris_center_y = result.iris_geometry.ellipse.center_y if result.iris_geometry.ellipse is not None else result.iris_geometry.center_y
        expected_x = (iris_center_x - reference.center_x) / (reference.major_axis / 2.0)
        expected_y = (iris_center_y - reference.center_y) / (reference.minor_axis / 2.0)

        self.assertAlmostEqual(feature_x, expected_x, places=5)
        self.assertAlmostEqual(feature_y, expected_y, places=5)

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
        self.assertIn("insufficient valid frames", session.failure_reason)

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
