import json
import socket
import time

import pytest

from sundew_core_sdk.config import SDKConfig
from sundew_core_sdk.controller import AdaptiveGateController
from sundew_core_sdk.ipc.adapter import IPCAdapter
from sundew_core_sdk.ipc.transport import IPCServer, IPCServerConfig
from sundew_core_sdk.metrics import MetricsTracker


def wait_for_address(server: IPCServer, timeout: float = 1.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if server.address:
            return server.address
        time.sleep(0.01)
    raise TimeoutError("Server did not expose address")


def test_transport_roundtrip_tcp():
    adapter = IPCAdapter(
        controller=AdaptiveGateController(SDKConfig()),
        tracker=MetricsTracker(),
    )
    adapter.controller.load_native()
    server = IPCServer(
        adapter,
        IPCServerConfig(socket_path=None, host="127.0.0.1", port=0),
    )
    server.start()

    try:
        host, port = wait_for_address(server)
        payload = json.dumps(
            {
                "type": "score_event",
                "event": {
                    "sequence": 9,
                    "features": [
                        {"key": "glucose_mgdl", "value": 135.0},
                        {"key": "roc_mgdl_min", "value": -1.2},
                    ],
                },
            }
        ).encode()
        with socket.create_connection((host, port)) as client:
            client.sendall(payload)
            response = json.loads(client.recv(4096).decode())
        assert response["sequence"] == 9
        assert "should_activate" in response
    finally:
        server.stop()
