"""Utility script to compile Sundew IPC protobufs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTO = ROOT / 'docs' / 'sdk' / 'ipc' / 'sundew_ipc_v1.proto'
OUT_DIR = ROOT / 'src'


def main() -> int:
    if not PROTO.exists():
        print(f"Proto file not found: {PROTO}", file=sys.stderr)
        return 1
    cmd = [
        sys.executable,
        '-m',
        'grpc_tools.protoc',
        f'--proto_path={PROTO.parent}',
        f'--python_out={OUT_DIR}',
        f'--grpc_python_out={OUT_DIR}',
        PROTO.name,
    ]
    print(' '.join(cmd))
    try:
        subprocess.check_call(cmd, cwd=PROTO.parent)
    except FileNotFoundError:
        print('grpc_tools not installed; run `pip install grpcio-tools`.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
