#!/usr/bin/env python3
"""Run the WASS2S seasonal PRCP forecast."""

from wass2s.cli import run_forecast


def main(argv=None):
    return run_forecast("PRCP", argv)


if __name__ == "__main__":
    main()
