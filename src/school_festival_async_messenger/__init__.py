import argparse
import base64
import binascii
import ipaddress
import logging
import shlex
import socket as network
import sys
from contextlib import ExitStack

import cv2
import numpy as np
import zmq

logger = logging.getLogger(__name__)


def encode_frame(frame: np.ndarray, quality: int = 85) -> bytes:
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
    except binascii.Error, ValueError, cv2.error:
        return None
    return frame


def _log_host_endpoint(endpoint: str) -> None:
    logger.info("配信開始: %s", endpoint)
    host_part, _, port = endpoint.rpartition(":")
    if host_part in ("tcp://*", "tcp://0.0.0.0"):
        candidates = set()
        try:
            for result in network.getaddrinfo(network.gethostname(), None, network.AF_INET, network.SOCK_STREAM):
                address = ipaddress.IPv4Address(result[4][0])
                if not (address.is_loopback or address.is_unspecified or address.is_multicast):
                    candidates.add(str(address))
        except OSError as error:
            logger.warning("ローカル IPv4 候補を取得できません: %s", error)
        if candidates:
            logger.info("接続先候補 (LAN 到達性は保証されません。IP・経路・ファイアウォールを確認):")
            endpoints = [f"tcp://{address}:{port}" for address in sorted(candidates)]
        else:
            logger.info("LAN 接続先候補なし。次の接続例は同じマシン専用です:")
            endpoints = [f"tcp://127.0.0.1:{port}"]
    else:
        endpoints = [endpoint]
    for target in endpoints:
        logger.info("受信例: uv run school-festival-async-messenger client --connect %s", shlex.quote(target))


def _close_host_preview() -> None:
    try:
        cv2.destroyAllWindows()
    except cv2.error as error:
        logger.warning("プレビューの終了処理に失敗しました: %s", error)


def host(camera_index: int, quality: int, address: str, *, preview: bool = False) -> int:
    try:
        with ExitStack() as cleanup:
            camera = cv2.VideoCapture(camera_index)
            cleanup.callback(camera.release)
            if not camera.isOpened():
                logger.error("カメラ %s を開けません。接続と権限を確認してください。", camera_index)
                return 1
            context = zmq.Context()
            cleanup.callback(context.term)
            socket = context.socket(zmq.PUB)
            cleanup.callback(socket.close, linger=0)
            socket.setsockopt(zmq.LINGER, 0)
            socket.bind(address)
            _log_host_endpoint(socket.getsockopt_string(zmq.LAST_ENDPOINT))
            if preview:
                cleanup.callback(_close_host_preview)
                cv2.namedWindow("Host preview - q / Esc to quit", cv2.WINDOW_AUTOSIZE)
                logger.info("プレビュー: ウィンドウで q / Esc、またはターミナルで Ctrl-C を押すと配信終了")
            while True:
                success, frame = camera.read()
                if not success:
                    logger.error("カメラからフレームを読み取れません。")
                    return 1
                payload = encode_frame(frame, quality)
                socket.send(payload)
                if preview:
                    decoded = decode_frame(payload)
                    if decoded is not None:
                        cv2.imshow("Host preview - q / Esc to quit", decoded)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        return 0
    except zmq.ZMQError, ValueError, cv2.error:
        logger.exception("配信を開始/継続できません。")
        return 1
    except KeyboardInterrupt:
        return 0


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
    host_parser.add_argument(
        "--quality",
        type=int,
        choices=range(1, 101),
        default=85,
        help="JPEG 品質 1〜100 (既定: 85)",
    )
    host_parser.add_argument("--bind", default="tcp://*:5555", help="配信アドレス (既定: tcp://*:5555)")
    host_parser.add_argument("--preview", action="store_true", help="送信映像を表示 (GUI 必須、q / Esc で配信終了)")
    client_parser = commands.add_parser("client", help="映像を受信して表示")
    client_parser.add_argument("--connect", default="tcp://127.0.0.1:5555", help="接続先 (既定: tcp://127.0.0.1:5555)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.command == "host":
        return host(args.camera, args.quality, args.bind, preview=args.preview)
    return client(args.connect)


if __name__ == "__main__":
    raise SystemExit(main())
