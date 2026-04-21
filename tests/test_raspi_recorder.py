import unittest
from unittest import mock

import numpy as np

from eyetrack.raspi.overlay import compose_recording_overlay
from eyetrack.raspi.recorder import build_recording_pipeline_launch, resolve_recording_h264_encoder_name


class RaspberryPiRecorderTests(unittest.TestCase):
    def test_compose_recording_overlay_draws_hollow_circle_at_screen_uv(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        output = compose_recording_overlay(frame, screen_uv=(0.5, 0.5), tracking_valid=True)

        self.assertEqual(output.shape, frame.shape)
        self.assertTrue(np.array_equal(output[50, 50], np.zeros(3, dtype=np.uint8)))
        self.assertTrue(np.array_equal(output[50, 64], np.array([0, 255, 0], dtype=np.uint8)))

    def test_compose_recording_overlay_skips_when_screen_uv_missing(self) -> None:
        frame = np.zeros((48, 64, 3), dtype=np.uint8)

        output = compose_recording_overlay(frame, screen_uv=None, tracking_valid=False)

        self.assertTrue(np.array_equal(output, frame))

    def test_build_recording_pipeline_launch_uses_matroska_filesink(self) -> None:
        launch = build_recording_pipeline_launch(
            width=1280,
            height=720,
            fps=30,
            encoder_fragment="v4l2h264enc ! h264parse",
            output_path="/tmp/fpv_gaze_test.mkv",
        )

        self.assertIn("matroskamux", launch)
        self.assertIn("filesink", launch)
        self.assertNotIn("rtph264pay", launch)
        self.assertNotIn("RTSP", launch)

    def test_resolve_recording_h264_encoder_name_rejects_x264_only(self) -> None:
        fake_gst = mock.Mock()
        fake_gst.ElementFactory.find.side_effect = lambda name: object() if name == "x264enc" else None

        with self.assertRaisesRegex(RuntimeError, "x264enc software fallback is disabled"):
            resolve_recording_h264_encoder_name(fake_gst)


if __name__ == "__main__":
    unittest.main()
