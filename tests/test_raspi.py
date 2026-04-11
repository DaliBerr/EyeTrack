import unittest

import numpy as np

from eyetrack.raspi.camera import convert_fpv_frame_to_bgr, prepare_eye_tensor_from_yuv420
from eyetrack.raspi.sync import compute_frame_age_ms, is_frame_stale, pair_eye_and_fpv_frames, resolve_frame_timestamp_ns
from eyetrack.raspi.types import CameraFrame


class RaspberryPiRuntimeTests(unittest.TestCase):
    def test_prepare_eye_tensor_from_yuv420_uses_y_plane_without_resize(self) -> None:
        height = 4
        width = 6
        y_plane = np.arange(height * width, dtype=np.uint8).reshape(height, width)
        uv_plane = np.zeros((height // 2, width), dtype=np.uint8)
        frame = np.vstack([y_plane, uv_plane])

        preview, tensor = prepare_eye_tensor_from_yuv420(frame, input_width=width, input_height=height)

        self.assertEqual(preview.shape, (height, width))
        self.assertTrue(np.array_equal(preview, y_plane))
        self.assertEqual(tensor.shape, (1, 1, height, width))
        self.assertAlmostEqual(float(tensor[0, 0, 0, 1]), float(y_plane[0, 1]) / 255.0, places=6)

    def test_prepare_eye_tensor_from_yuv420_rejects_mismatched_shape(self) -> None:
        frame = np.zeros((9, 5), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "尺寸异常|模型输入不匹配"):
            prepare_eye_tensor_from_yuv420(frame, input_width=6, input_height=4)

    def test_pair_eye_and_fpv_frames_marks_sync_stale(self) -> None:
        eye_frame = CameraFrame(
            camera_id=0,
            frame=np.zeros((2, 2), dtype=np.uint8),
            timestamp_ns=1_000_000_000,
            captured_monotonic_ns=1_500_000_000,
            frame_index=1,
        )
        fpv_frame = CameraFrame(
            camera_id=1,
            frame=np.zeros((2, 2, 3), dtype=np.uint8),
            timestamp_ns=1_090_000_000,
            captured_monotonic_ns=1_550_000_000,
            frame_index=2,
        )

        pair = pair_eye_and_fpv_frames(eye_frame=eye_frame, fpv_frame=fpv_frame, tolerance_ms=40.0)

        self.assertAlmostEqual(pair.skew_ms, 90.0, places=4)
        self.assertTrue(pair.sync_stale)

    def test_resolve_frame_timestamp_ns_prefers_sensor_timestamp(self) -> None:
        timestamp_ns = resolve_frame_timestamp_ns({"SensorTimestamp": 123456789}, fallback_monotonic_ns=999)
        self.assertEqual(timestamp_ns, 123456789)

    def test_is_frame_stale_uses_capture_monotonic_timestamp(self) -> None:
        frame = CameraFrame(
            camera_id=0,
            frame=np.zeros((2, 2), dtype=np.uint8),
            timestamp_ns=100,
            captured_monotonic_ns=1_000_000_000,
            frame_index=0,
        )

        self.assertAlmostEqual(compute_frame_age_ms(frame, now_ns=1_120_000_000), 120.0, places=4)
        self.assertTrue(is_frame_stale(frame, stale_after_ms=100.0, now_ns=1_120_000_000))
        self.assertFalse(is_frame_stale(frame, stale_after_ms=200.0, now_ns=1_120_000_000))

    def test_convert_fpv_frame_to_bgr_supports_rgb888(self) -> None:
        rgb = np.zeros((1, 1, 3), dtype=np.uint8)
        rgb[0, 0] = [10, 20, 30]

        bgr = convert_fpv_frame_to_bgr(rgb, pixel_format="RGB888")

        self.assertEqual(bgr.shape, (1, 1, 3))
        self.assertTrue(np.array_equal(bgr[0, 0], np.array([30, 20, 10], dtype=np.uint8)))


if __name__ == "__main__":
    unittest.main()
