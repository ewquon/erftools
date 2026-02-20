"""Thin entrypoint: download ERA5 data and write ERF initial condition files.

Usage
-----
Single-timestep reanalysis::

    python3 WriteICFromERA5Data.py input_for_Laura

Multi-timestep forecast (MPI-parallelised)::

    srun -n 32 python3 WriteICFromERA5Data.py input_for_Laura \\
        --do_forecast=true --forecast_time_hours=72 --interval_hours=3

See ``erftools.hindcast.ERA5Hindcast`` for full workflow details.
"""
import argparse
import sys
import time

from mpi4py import MPI

from erftools.hindcast import ERA5Hindcast


if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    start_time = time.time()

    if len(sys.argv) == 1:
        print("Usage: python3 WriteICFromERA5Data.py <input_filename> [--do_forecast=true]")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Download and process ERA5 data.")
    parser.add_argument("input_filename", help="Input filename, e.g. input_for_Laura")
    parser.add_argument(
        "--do_forecast", type=lambda x: x.lower() == "true", default=False,
        help="Set to true to download forecast data",
    )
    parser.add_argument("--forecast_time_hours", type=int,
                        help="Forecast length in hours, e.g. 72")
    parser.add_argument("--interval_hours", type=int,
                        help="Forecast interval in hours, e.g. 3")

    args = parser.parse_args()

    if args.do_forecast and (args.forecast_time_hours is None or args.interval_hours is None):
        parser.error("--do_forecast requires --forecast_time_hours and --interval_hours")

    hindcast = ERA5Hindcast(
        args.input_filename,
        do_forecast=args.do_forecast,
        forecast_time_hours=args.forecast_time_hours,
        interval_hours=args.interval_hours,
    )
    hindcast.run(comm=comm)

    end_time = time.time()
    elapsed = end_time - start_time
    max_elapsed = comm.reduce(elapsed, op=MPI.MAX, root=0)
    if rank == 0:
        print(f"Total runtime (wall-clock, across ranks): {max_elapsed:.2f} seconds", flush=True)
