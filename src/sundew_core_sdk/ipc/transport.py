
"""Socket-based transport prototype for the IPC adapter."""

from __future__ import annotations

import json
import os
import socket
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

from .adapter import IPCAdapter

try:
    from sundew_ipc_v1_pb2 import FeatureKV, ScoreEvent
except ModuleNotFoundError:  # pragma: no cover - optional binding
    FeatureKV = ScoreEvent = None


@dataclass
class IPCServerConfig:
    socket_path: Optional[str] = None
    host: str = "127.0.0.1"
    port: int = 0


class IPCServer:
    def __init__(self, adapter: IPCAdapter, config: IPCServerConfig) -> None:
        self.adapter = adapter
        self.config = config
        self._thread: Optional[threading.Thread] = None
        self._sock: Optional[socket.socket] = None
        self._stop = threading.Event()
        self._address: Optional[Tuple[str, int]] = None
        self._use_unix = bool(
            hasattr(socket, "AF_UNIX") and config.socket_path
        )

    @property
    def address(self) -> Optional[Tuple[str, int]]:
        return self._address

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
        if self._thread:
            self._thread.join(timeout=2)
        if self._use_unix and self.config.socket_path:
            try:
                os.remove(self.config.socket_path)
            except FileNotFoundError:
                pass

    def _serve(self) -> None:
        if self._use_unix:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)  # type: ignore[attr-defined]
            path = self.config.socket_path or ""
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            sock.bind(path)
        else:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind((self.config.host, self.config.port))
            self._address = sock.getsockname()
        self._sock = sock
        try:
            sock.listen(1)
            while not self._stop.is_set():
                try:
                    conn, _ = sock.accept()
                except OSError:
                    break
                with conn:
                    data = conn.recv(4096)
                    if not data:
                        continue
                    payload = json.loads(data.decode())
                    message_type = payload.get("type")
                    if message_type == "score_event":
                        event_obj = self._decode_event(payload["event"])
                        decision = self.adapter.handle_score_event(event_obj)
                        response = json.dumps(
                            {
                                "sequence": getattr(decision, "sequence", 0),
                                "should_activate": getattr(
                                    decision, "should_activate", False
                                ),
                            }
                        )
                        conn.sendall(response.encode())
        finally:
            try:
                sock.close()
            except OSError:
                pass

    def _decode_event(self, payload: dict):
        if ScoreEvent is not None:
            event = ScoreEvent(sequence=payload.get("sequence", 0))
            for item in payload.get("features", []):
                feature = FeatureKV(key=item["key"], value=item["value"])
                event.features.append(feature)
            return event

        class _Feature:
            def __init__(self, key: str, value: float) -> None:
                self.key = key
                self.value = value

        class _Event:
            def __init__(self, data: dict) -> None:
                self.sequence = data.get("sequence", 0)
                self.features = [
                    _Feature(item["key"], item["value"])
                    for item in data.get("features", [])
                ]

        return _Event(payload)
