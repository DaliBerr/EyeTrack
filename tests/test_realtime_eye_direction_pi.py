import unittest
from unittest import mock

from eyetrack.gaze import begin_calibration_session
from realtime_eye_direction_pi import stop_active_recording, validate_recording_start_request


class RealtimePiRecordingControlTests(unittest.TestCase):
    def test_validate_recording_start_request_rejects_uncalibrated_session(self) -> None:
        session = begin_calibration_session(
            now_ms=0.0,
            settle_ms=500,
            capture_ms=1000,
            min_valid_frames=5,
            point_pattern="nine",
        )

        reason = validate_recording_start_request(
            eye_only_mode=False,
            calibration_session=session,
            fpv_frame_available=True,
            recording_active=False,
        )

        self.assertEqual(reason, "Recording requires a completed calibration.")

    def test_validate_recording_start_request_requires_fpv_frame(self) -> None:
        session = begin_calibration_session(
            now_ms=0.0,
            settle_ms=500,
            capture_ms=1000,
            min_valid_frames=5,
            point_pattern="nine",
        )
        session.state = "completed"
        session.model = object()

        reason = validate_recording_start_request(
            eye_only_mode=False,
            calibration_session=session,
            fpv_frame_available=False,
            recording_active=False,
        )

        self.assertEqual(reason, "Cannot start recording because no FPV frame is available yet.")

    def test_validate_recording_start_request_allows_completed_calibration(self) -> None:
        session = begin_calibration_session(
            now_ms=0.0,
            settle_ms=500,
            capture_ms=1000,
            min_valid_frames=5,
            point_pattern="nine",
        )
        session.state = "completed"
        session.model = object()

        reason = validate_recording_start_request(
            eye_only_mode=False,
            calibration_session=session,
            fpv_frame_available=True,
            recording_active=False,
        )

        self.assertIsNone(reason)

    def test_stop_active_recording_calls_recorder_stop(self) -> None:
        recorder = mock.Mock()
        type(recorder).is_recording = mock.PropertyMock(return_value=True)
        recorder.stop.return_value = "/tmp/fpv_gaze_test.mkv"

        output_path, error = stop_active_recording(recorder)

        self.assertEqual(output_path, "/tmp/fpv_gaze_test.mkv")
        self.assertIsNone(error)
        recorder.stop.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
