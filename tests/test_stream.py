import base64
import time
import unittest

import numpy as np
import zmq

from school_festival_async_messenger import decode_frame, encode_frame


class FrameCodecTests(unittest.TestCase):
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
