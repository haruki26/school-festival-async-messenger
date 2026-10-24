import argparse
import base64
import binascii
import sys

import cv2
import numpy as np
import zmq


def encode_frame(frame: np.ndarray, quality: int = 80) -> bytes:
    if not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be between 1 and 100")
    success, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise ValueError("could not encode camera frame as JPEG")
    return base64.b64encode(jpeg.tobytes())


def decode_frame(payload: bytes) -> np.ndarray | None:
    try:
        jpeg = base64.b64decode(payload, validate=True)
        if not jpeg:
            return None
        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    except (binascii.Error, ValueError, cv2.error):
        return None
    return frame


def host(camera_index: int, quality: int, address: str) -> int:
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        camera.release()
        print(f"カメラ {camera_index} を開けません。接続と権限を確認してください。", file=sys.stderr)
        return 1

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.LINGER, 0)
    try:
        socket.bind(address)
        while True:
            success, frame = camera.read()
            if not success:
                print("カメラからフレームを読み取れません。", file=sys.stderr)
                return 1
            socket.send(encode_frame(frame, quality))
    except (zmq.ZMQError, ValueError) as error:
        print(f"配信を開始/継続できません: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        socket.close(linger=0)
        context.term()
        camera.release()


def client(address: str) -> int:
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    try:
        socket.connect(address)
        cv2.namedWindow("Camera stream", cv2.WINDOW_AUTOSIZE)
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        while True:
            if dict(poller.poll(50)).get(socket, 0) & zmq.POLLIN:
                frame = decode_frame(socket.recv())
                if frame is not None:
                    cv2.imshow("Camera stream", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                return 0
    except zmq.ZMQError as error:
        print(f"通信エラー: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        socket.close(linger=0)
        context.term()
        cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="学校祭向け ZMQ カメラ映像ストリーマー")
    commands = parser.add_subparsers(dest="command", required=True)
    host_parser = commands.add_parser("host", help="カメラ映像を配信")
    host_parser.add_argument("--camera", type=int, default=0, help="カメラ番号 (既定: 0)")
    host_parser.add_argument("--quality", type=int, choices=range(1, 101), default=80, help="JPEG 品質 1〜100 (既定: 80)")
    host_parser.add_argument("--bind", default="tcp://*:5555", help="配信アドレス (既定: tcp://*:5555)")
    client_parser = commands.add_parser("client", help="映像を受信して表示")
    client_parser.add_argument("--connect", default="tcp://127.0.0.1:5555", help="接続先 (既定: tcp://127.0.0.1:5555)")
    args = parser.parse_args(argv)
    if args.command == "host":
        return host(args.camera, args.quality, args.bind)
    return client(args.connect)


if __name__ == "__main__":
    raise SystemExit(main())
