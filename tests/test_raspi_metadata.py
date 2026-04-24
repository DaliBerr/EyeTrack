import unittest

from eyetrack.gaze import begin_calibration_session
from eyetrack.raspi.metadata import build_gaze_metadata_packet


class RaspberryPiMetadataTests(unittest.TestCase):
    def test_build_gaze_metadata_packet_includes_calibration_target(self) -> None:
        session = begin_calibration_session(
            now_ms=0.0,
            settle_ms=500,
            capture_ms=1000,
            min_valid_frames=15,
            margin=0.1,
        )

        packet = build_gaze_metadata_packet(
            eye_timestamp_ns=100,
            fpv_timestamp_ns=200,
            screen_uv=(0.4, 0.6),
            fpv_output_mode="record",
            recording_active=True,
            tracking_valid=True,
            feature_valid=True,
            feature_mode="iris_only",
            calibration_session=session,
            sync_skew_ms=12.5,
            sync_stale=False,
            eye_stale=False,
            fpv_stale=False,
            fps=29.8,
            inference_ms=8.2,
            status_message="ok",
        )

        payload = packet.to_dict()
        self.assertEqual(payload["screen_uv"], [0.4, 0.6])
        self.assertEqual(payload["calibration_target_uv"], [0.1, 0.1])
        self.assertEqual(payload["fpv_output_mode"], "record")
        self.assertTrue(payload["recording_active"])
        self.assertEqual(payload["calibration_state"], "settling")
        self.assertFalse(payload["calibrated"])

    def test_gaze_metadata_json_line_is_compact(self) -> None:
        packet = build_gaze_metadata_packet(
            eye_timestamp_ns=None,
            fpv_timestamp_ns=None,
            screen_uv=None,
            fpv_output_mode="rtsp",
            recording_active=False,
            tracking_valid=False,
            feature_valid=False,
            feature_mode="iris_only",
            calibration_session=None,
            sync_skew_ms=None,
            sync_stale=True,
            eye_stale=True,
            fpv_stale=False,
            fps=0.0,
            inference_ms=None,
            status_message="waiting",
        )

        line = packet.to_json_line()
        self.assertIn('"calibration_state":"idle"', line)
        self.assertIn('"fpv_output_mode":"rtsp"', line)
        self.assertIn('"recording_active":false', line)
        self.assertIn('"sync_stale":true', line)
        self.assertNotIn("\n", line)


if __name__ == "__main__":
    unittest.main()
