
"""Compare baseline vs gated power CSV logs and compute savings."""

from __future__ import annotations

import argparse
import csv
from statistics import mean


def load_power(path: str) -> float:
    with open(path, newline='') as fh:
        reader = csv.DictReader(fh)
        power = [float(row['power_mw']) for row in reader]
    if not power:
        raise ValueError(f'No power samples found in {path}')
    return mean(power)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, help='CSV log from ungated run')
    parser.add_argument('--gated', required=True, help='CSV log from Sundew gated run')
    args = parser.parse_args()

    baseline_mw = load_power(args.baseline)
    gated_mw = load_power(args.gated)
    if baseline_mw <= 0:
        raise ValueError('Baseline power must be > 0')
    savings = 1.0 - (gated_mw / baseline_mw)
    print(f'Baseline power (mW): {baseline_mw:.2f}')
    print(f'Gated power (mW): {gated_mw:.2f}')
    print(f'Estimated savings: {savings:.2%}')


if __name__ == '__main__':
    main()
