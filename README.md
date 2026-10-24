# 学校祭 非同期カメラ映像ストリーマー

OpenCV のカメラ映像を JPEG/Base64 にして ZeroMQ PUB/SUB で配信します。送受信する両方のマシンに Python 3.14 以降と `uv` が必要です。

## 配信側

カメラを接続し、リポジトリで依存関係を準備してから起動します。

```sh
uv sync
uv run school-festival-async-messenger host
```

既定ではカメラ番号 `0`、JPEG 品質 `80`、`tcp://*:5555` で配信します。必要に応じて変更できます。

```sh
uv run school-festival-async-messenger host --camera 1 --quality 70 --bind tcp://*:5555
```

## 受信側

受信側にもリポジトリと依存関係が必要です。`HOST_IP` を配信側マシンの LAN 内 IP アドレスに置き換えます。

```sh
uv sync
uv run school-festival-async-messenger client --connect tcp://HOST_IP:5555
```

受信側には GUI デスクトップ環境が必要です（OpenCV のネイティブウィンドウを表示します）。映像ウィンドウで `q` または Esc、あるいはターミナルで Ctrl-C を押すと終了します。

## 注意事項

- 配信側のファイアウォールで TCP ポート 5555 への接続を許可してください。別ポートを使う場合は、配信側の `--bind` と受信側の `--connect` を合わせます。
- この通信は暗号化・認証されません。信頼できる LAN 内だけで使用し、インターネットに公開しないでください。
- JPEG は非可逆圧縮です。PUB/SUB は最新フレームを優先して古いフレームを捨てるため、接続直後や回線混雑時にはフレームが欠落します。Base64 により JPEG データ量は約 4/3 になります。
- カメラ利用権限と、OpenCV が利用できるカメラデバイスが必要です。
