
"""Plot power vs. time from an exported benchmark JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(path: str) -> int:
    source = Path(path)
    data = json.loads(source.read_text(encoding='utf-8'))
    events = data.get('events', [])
    if not events:
        print('No events found in export; nothing to plot.', file=sys.stderr)
        return 1
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print('matplotlib not installed; install it to visualize results.', file=sys.stderr)
        return 1
    except Exception as exc:
        print(f'Failed to import matplotlib dependencies: {exc}', file=sys.stderr)
        return 1
    timestamps = [event['timestamp'] for event in events]
    power = [event['power_w'] for event in events]
    activated = [event['activated'] for event in events]
    plt.figure(figsize=(10, 4))
    plt.plot(timestamps, power, label='Power (W)')
    plt.scatter(
        [t for t, a in zip(timestamps, activated) if a],
        [p for p, a in zip(power, activated) if a],
        s=10,
        c='red',
        label='Activated',
    )
    plt.title(f"Sundew profile={data.get('profile', 'unknown')}")
    plt.xlabel('Event')
    plt.ylabel('Power (W)')
    plt.legend()
    plt.tight_layout()
    output = source.with_suffix('.png')
    plt.savefig(output)
    print(f'Wrote {output}')
    return 0


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python benchmarks/power/plot_export.py <export.json>', file=sys.stderr)
        raise SystemExit(1)
    raise SystemExit(main(sys.argv[1]))
