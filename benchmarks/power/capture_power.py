"""Skeleton for capturing power telemetry via INA sensors."""

from __future__ import annotations

import argparse
import csv
import time

try:
    import adafruit_ina219  # type: ignore
    import board  # type: ignore
    import busio  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    adafruit_ina219 = board = busio = None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration', type=float, default=60.0)
    parser.add_argument('--interval', type=float, default=0.5)
    parser.add_argument('--output', default='power_log.csv')
    args = parser.parse_args()

    if adafruit_ina219 is None:
        raise RuntimeError('INA219 library not installed')

    i2c = busio.I2C(board.SCL, board.SDA)
    sensor = adafruit_ina219.INA219(i2c)

    with open(args.output, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['timestamp', 'voltage_v', 'current_ma', 'power_mw'])
        end = time.time() + args.duration
        while time.time() < end:
            voltage = sensor.bus_voltage
            current = sensor.current
            power = sensor.power
            writer.writerow([time.time(), voltage, current, power])
            time.sleep(args.interval)


if __name__ == '__main__':
    main()
