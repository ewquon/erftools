"""Thin entrypoint: download GFS data and write ERF initial condition files.

Usage
-----
Single-timestep analysis::

    python3 WriteICFromGFSData.py input_for_Laura

Multi-timestep forecast series::

    python3 WriteICFromGFSData.py input_for_Laura --do_forecast=True

See ``erftools.hindcast.GFSHindcast`` for full workflow details.
"""
import argparse
import sys

from erftools.hindcast import GFSHindcast


if __name__ == "__main__":

    if len(sys.argv) == 1:
        print("Usage: python3 WriteICFromGFSData.py <input_filename> [--do_forecast=True]")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Download and process GFS data.")
    parser.add_argument("input_filename", help="Input filename, e.g. input_for_Laura")
    parser.add_argument(
        "--do_forecast", type=lambda x: x.lower() == "true", default=False,
        help="Set to True to download forecast data series",
    )

    args = parser.parse_args()

    hindcast = GFSHindcast(args.input_filename, do_forecast=args.do_forecast)
    hindcast.run()
