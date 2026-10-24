import base64
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import zmq

import school_festival_async_messenger as stream
from school_festival_async_messenger import decode_frame, encode_frame


class FrameCodecTests(unittest.TestCase):
    def test_quality_default_and_override(self) -> None:
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        for quality in (None, 61):
            with self.subTest(quality=quality), patch.object(cv2, "imencode", wraps=cv2.imencode) as encode:
                if quality is None:
                    encode_frame(frame)
                else:
                    encode_frame(frame, quality)
                assert encode.call_args.args[2] == [cv2.IMWRITE_JPEG_QUALITY, 85 if quality is None else quality]

    def test_jpeg_base64_round_trip(self) -> None:
        frame = np.zeros((32, 48, 3), dtype=np.uint8)
        frame[:, :, 2] = 220
        decoded = decode_frame(encode_frame(frame))
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded.shape, frame.shape)
        self.assertGreater(int(decoded[:, :, 2].mean()), 180)

    def test_malformed_payloads_are_ignored(self) -> None:
        for payload in (b"%%%", b"", base64.b64encode(b"not an image")):
            with self.subTest(payload=payload):
                self.assertIsNone(decode_frame(payload))


class HostTests(unittest.TestCase):
    def setUp(self) -> None:
        capture = self.enterContext(patch.object(cv2, "VideoCapture"))
        self.camera = capture.return_value
        self.camera.isOpened.return_value = True
        self.camera.read.side_effect = KeyboardInterrupt
        context = self.enterContext(patch.object(zmq, "Context"))
        self.context = context.return_value
        self.socket = self.context.socket.return_value
        self.socket.getsockopt_string.return_value = "tcp://192.168.1.20:6001"
        self.window = self.enterContext(patch.object(cv2, "namedWindow"))
        self.show = self.enterContext(patch.object(cv2, "imshow"))
        self.key = self.enterContext(patch.object(cv2, "waitKey", return_value=ord("q")))
        self.destroy = self.enterContext(patch.object(cv2, "destroyAllWindows"))
        self.discovery = self.enterContext(patch.object(stream.network, "getaddrinfo", return_value=[]))
        self.enterContext(patch.object(stream.network, "gethostname", return_value="festival-host"))

    def assert_released(self) -> None:
        self.camera.release.assert_called_once_with()
        self.socket.close.assert_called_once_with(linger=0)
        self.context.term.assert_called_once_with()

    def test_cli_defaults_and_overrides(self) -> None:
        with (
            patch.object(stream, "host", return_value=0) as host,
            patch.object(stream.logging, "basicConfig") as configure_logging,
        ):
            assert stream.main(["host"]) == 0
            host.assert_called_once_with(0, 85, "tcp://*:5555", preview=False)
            configure_logging.assert_called_with(level=stream.logging.INFO, format="%(levelname)s: %(message)s")
            host.reset_mock()
            assert stream.main(["host", "--quality", "61", "--preview", "--bind", "tcp://*:6001"]) == 0
            host.assert_called_once_with(0, 61, "tcp://*:6001", preview=True)

    def test_preview_off_sends_without_any_gui_calls(self) -> None:
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        self.camera.read.side_effect = [(True, frame), KeyboardInterrupt]
        assert stream.host(0, 85, "tcp://*:6001") == 0
        assert self.socket.send.call_args.args[0] == encode_frame(frame)
        for gui in (self.window, self.show, self.key, self.destroy):
            gui.assert_not_called()
        self.assert_released()

    def test_preview_decodes_sent_payload_and_quit_keys_release_resources(self) -> None:
        frame = np.arange(192, dtype=np.uint8).reshape((8, 8, 3))
        for key in (ord("q"), 27):
            with self.subTest(key=key), patch.object(stream, "decode_frame", wraps=decode_frame) as decode:
                self.camera.read.side_effect = None
                self.camera.read.return_value = (True, frame)
                self.key.return_value = key
                assert stream.host(0, 61, "tcp://*:6001", preview=True) == 0
                payload = self.socket.send.call_args.args[0]
                assert payload == encode_frame(frame, 61)
                decode.assert_called_once_with(payload)
                np.testing.assert_array_equal(self.show.call_args.args[1], decode_frame(payload))
                self.assert_released()
                self.destroy.assert_called_once_with()
                for resource in (self.camera, self.socket, self.context, self.destroy):
                    resource.reset_mock()

    def test_preview_keyboard_interrupt_releases_resources(self) -> None:
        assert stream.host(0, 85, "tcp://*:6001", preview=True) == 0
        self.assert_released()
        self.destroy.assert_called_once_with()

    def test_opencv_errors_release_resources(self) -> None:
        self.camera.read.side_effect = None
        self.camera.read.return_value = (True, np.zeros((8, 8, 3), dtype=np.uint8))
        for operation in (self.window, self.show, self.key):
            with self.subTest(operation=operation), self.assertLogs(stream.logger, level="ERROR"):
                operation.side_effect = cv2.error("GUI unavailable")
                assert stream.host(0, 85, "tcp://*:6001", preview=True) == 1
                self.assert_released()
                self.destroy.assert_called_once_with()
                operation.side_effect = None
                for resource in (self.camera, self.socket, self.context, self.destroy):
                    resource.reset_mock()

    def test_encode_error_without_preview_never_calls_gui(self) -> None:
        self.camera.read.side_effect = None
        self.camera.read.return_value = (True, np.zeros((8, 8, 3), dtype=np.uint8))
        with (
            patch.object(cv2, "imencode", side_effect=cv2.error("encode failed")),
            self.assertLogs(stream.logger, level="ERROR"),
        ):
            assert stream.host(0, 85, "tcp://*:6001") == 1
        self.assert_released()
        for gui in (self.window, self.show, self.key, self.destroy):
            gui.assert_not_called()

    def test_preview_cleanup_error_does_not_mask_exit(self) -> None:
        self.destroy.side_effect = cv2.error("GUI unavailable")
        with self.assertLogs(stream.logger, level="WARNING"):
            assert stream.host(0, 85, "tcp://*:6001", preview=True) == 0
        self.assert_released()

    def test_startup_logs_actual_explicit_endpoint(self) -> None:
        with self.assertLogs(stream.logger, level="INFO") as logs:
            assert stream.host(0, 85, "tcp://192.168.1.20:*") == 0
        output = "\n".join(logs.output)
        assert "配信開始: tcp://192.168.1.20:6001" in output
        assert "client --connect tcp://192.168.1.20:6001" in output
        self.discovery.assert_not_called()
        self.socket.bind.assert_called_once_with("tcp://192.168.1.20:*")

    def test_wildcard_logs_deduplicated_ipv4_candidates_with_actual_port(self) -> None:
        self.discovery.return_value = [
            (stream.network.AF_INET, stream.network.SOCK_STREAM, 6, "", (address, 0))
            for address in ("192.168.1.20", "10.0.0.5", "192.168.1.20", "127.0.0.1")
        ]
        for endpoint in ("tcp://0.0.0.0:6001", "tcp://*:6001"):
            with self.subTest(endpoint=endpoint), self.assertLogs(stream.logger, level="INFO") as logs:
                self.socket.getsockopt_string.return_value = endpoint
                assert stream.host(0, 85, "tcp://*:6001") == 0
                output = "\n".join(logs.output)
                assert "LAN 到達性は保証されません" in output
                assert output.count("client --connect tcp://192.168.1.20:6001") == 1
                assert "client --connect tcp://10.0.0.5:6001" in output
                assert "client --connect tcp://0.0.0.0" not in output
                assert "client --connect tcp://*" not in output
        self.discovery.assert_called_with("festival-host", None, stream.network.AF_INET, stream.network.SOCK_STREAM)

    def test_discovery_failure_or_no_candidates_keeps_host_running_with_local_fallback(self) -> None:
        self.socket.getsockopt_string.return_value = "tcp://0.0.0.0:6001"
        for failure in (None, OSError("hostname unavailable")):
            with self.subTest(failure=failure), self.assertLogs(stream.logger, level="INFO") as logs:
                self.discovery.side_effect = failure
                assert stream.host(0, 85, "tcp://*:6001") == 0
                output = "\n".join(logs.output)
                assert "同じマシン専用" in output
                assert "client --connect tcp://127.0.0.1:6001" in output
                if failure is not None:
                    assert "取得できません" in output

    def test_ipc_endpoint_does_not_invent_tcp_address(self) -> None:
        self.socket.getsockopt_string.return_value = "ipc:///tmp/festival-camera"
        with self.assertLogs(stream.logger, level="INFO") as logs:
            assert stream.host(0, 85, "ipc:///tmp/festival-camera") == 0
        output = "\n".join(logs.output)
        assert "client --connect ipc:///tmp/festival-camera" in output
        assert "tcp://" not in output
        self.discovery.assert_not_called()

    def test_bind_failure_does_not_log_startup_and_releases_resources(self) -> None:
        self.socket.bind.side_effect = zmq.ZMQError("bind failed")
        with self.assertLogs(stream.logger, level="INFO") as logs:
            assert stream.host(0, 85, "tcp://*:6001") == 1
        assert "配信開始" not in "\n".join(logs.output)
        self.assert_released()

    def test_unavailable_camera_is_released_without_starting_socket(self) -> None:
        self.camera.isOpened.return_value = False
        with self.assertLogs(stream.logger, level="ERROR"):
            assert stream.host(0, 85, "tcp://*:6001") == 1
        self.camera.release.assert_called_once_with()
        self.context.socket.assert_not_called()
        self.destroy.assert_not_called()

    def test_camera_read_failure_releases_resources(self) -> None:
        self.camera.read.side_effect = None
        self.camera.read.return_value = (False, None)
        with self.assertLogs(stream.logger, level="ERROR"):
            assert stream.host(0, 85, "tcp://*:6001", preview=True) == 1
        self.assert_released()
        self.destroy.assert_called_once_with()

    def test_socket_creation_failure_releases_context_and_camera(self) -> None:
        self.context.socket.side_effect = zmq.ZMQError("socket failed")
        with self.assertLogs(stream.logger, level="ERROR"):
            assert stream.host(0, 85, "tcp://*:6001") == 1
        self.context.term.assert_called_once_with()
        self.camera.release.assert_called_once_with()


class SocketTests(unittest.TestCase):
    def test_loopback_pub_sub_round_trip(self) -> None:
        context = zmq.Context()
        publisher = context.socket(zmq.PUB)
        subscriber = context.socket(zmq.SUB)
        publisher.setsockopt(zmq.LINGER, 0)
        subscriber.setsockopt(zmq.LINGER, 0)
        subscriber.setsockopt(zmq.CONFLATE, 1)
        subscriber.setsockopt(zmq.SUBSCRIBE, b"")
        try:
            publisher.bind("tcp://127.0.0.1:*")
            subscriber.connect(publisher.getsockopt(zmq.LAST_ENDPOINT).decode())
            poller = zmq.Poller()
            poller.register(subscriber, zmq.POLLIN)
            frame = np.full((24, 40, 3), 120, dtype=np.uint8)
            payload = encode_frame(frame)
            deadline = time.monotonic() + 3
            received = None
            while time.monotonic() < deadline:
                publisher.send(payload)
                if dict(poller.poll(20)).get(subscriber, 0) & zmq.POLLIN:
                    received = decode_frame(subscriber.recv())
                    if received is not None:
                        break
            self.assertIsNotNone(received, "PUB/SUB did not deliver a valid frame before deadline")
            self.assertEqual(received.shape, frame.shape)
        finally:
            subscriber.close(linger=0)
            publisher.close(linger=0)
            context.term()


if __name__ == "__main__":
    unittest.main()
