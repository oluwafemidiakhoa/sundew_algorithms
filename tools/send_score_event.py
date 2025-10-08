
"""CLI utility to send a ScoreEvent JSON to the IPC server."""

from __future__ import annotations

import argparse
import json
import socket
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--sequence', type=int, default=0)
    parser.add_argument('--feature', action='append', default=[], help='key=value')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features: List[dict[str, float]] = []
    for item in args.feature:
        key, value = item.split('=', 1)
        features.append({'key': key, 'value': float(value)})
    payload = {
        'type': 'score_event',
        'event': {
            'sequence': args.sequence,
            'features': features,
        },
    }
    data = json.dumps(payload).encode()
    with socket.create_connection((args.host, args.port)) as client:
        client.sendall(data)
        print(client.recv(4096).decode())


if __name__ == '__main__':
    main()
