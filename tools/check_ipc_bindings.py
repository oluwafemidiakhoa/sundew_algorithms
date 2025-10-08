
"""CI check ensuring generated IPC bindings are importable."""

from __future__ import annotations

import sys

from sundew_core_sdk.ipc.bindings import load_grpc_module, load_proto_module


def main() -> int:
    proto = load_proto_module()
    grpc = load_grpc_module()
    missing = []
    if proto is None:
        missing.append('sundew_ipc_v1_pb2')
    if grpc is None:
        missing.append('sundew_ipc_v1_pb2_grpc')
    if missing:
        print(f"Missing IPC bindings: {', '.join(missing)}", file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
