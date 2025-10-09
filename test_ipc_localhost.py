"""Simple localhost IPC test - runs server and client in one script."""

import socket
import json
import time
import threading
from sundew_core_sdk.config import SDKConfig
from sundew_core_sdk.controller import AdaptiveGateController
from sundew_core_sdk.ipc.adapter import IPCAdapter
from sundew_core_sdk.ipc.transport import IPCServer, IPCServerConfig
from sundew_core_sdk.metrics import MetricsTracker


def start_server():
    """Start IPC server in background."""
    adapter = IPCAdapter(
        controller=AdaptiveGateController(SDKConfig(target_activation=0.15)),
        tracker=MetricsTracker()
    )
    adapter.controller.load_native()
    server = IPCServer(adapter, IPCServerConfig(port=8765))
    server.start()
    print("✓ IPC Server started on localhost:8765")
    time.sleep(30)  # Run for 30 seconds
    server.stop()


def send_event(feature_name, feature_value):
    """Send a test event to the server."""
    payload = json.dumps({
        "type": "score_event",
        "event": {
            "sequence": 1,
            "features": [{"key": feature_name, "value": feature_value}]
        }
    }).encode()

    try:
        with socket.create_connection(("127.0.0.1", 8765), timeout=5) as sock:
            sock.sendall(payload)
            response = sock.recv(4096).decode()
            result = json.loads(response)

            print(f"\n{'='*50}")
            print(f"Feature: {feature_name} = {feature_value}")
            print(f"Decision: {'✓ ACTIVATE' if result.get('should_activate') else '✗ SKIP'}")
            print(f"Sequence: {result.get('sequence')}")
            print('='*50)
            return result
    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    print("\n" + "="*60)
    print("Sundew SDK - Localhost IPC Test")
    print("="*60 + "\n")

    # Start server in background thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    # Wait for server to start
    print("Waiting for server to start...")
    time.sleep(2)

    # Send test events
    print("\nSending test events...\n")

    test_cases = [
        ("glucose_mgdl", 140.0),
        ("glucose_mgdl", 200.0),
        ("glucose_mgdl", 80.0),
        ("heart_rate", 120.0),
        ("temperature", 38.5),
    ]

    for feature, value in test_cases:
        send_event(feature, value)
        time.sleep(0.5)

    print("\n" + "="*60)
    print("✓ Localhost IPC test complete!")
    print("="*60 + "\n")

    print("The server will stop in ~25 seconds...")
    time.sleep(5)


if __name__ == "__main__":
    main()
