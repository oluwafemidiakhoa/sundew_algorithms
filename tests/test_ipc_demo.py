
import subprocess
import sys
from pathlib import Path


def test_ipc_demo_runs(tmp_path):
    env = {}
    result = subprocess.run(
        [sys.executable, 'examples/ipc_demo.py'],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert 'Gate decision:' in result.stdout
    assert 'Telemetry:' in result.stdout
