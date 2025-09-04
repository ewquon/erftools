from importlib import resources
import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from erftools.preprocessing import Download_ERA5_Data
from erftools.preprocessing import Download_ERA5_ForecastData
from erftools.preprocessing import ReadERA5_3DData

from erftools.utils.map import write_US_map_vtk


def read_user_input(filename):
    data = {}
    with open(filename, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.strip().split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                if key == 'area':
                    data[key] = [float(x) for x in value.split(',')]
                elif key == 'time':
                    data[key] = value
                else:
                    data[key] = int(value)
    return data


if __name__ == "__main__":

     # --- Parse arguments ---
    parser = argparse.ArgumentParser(description="Write USMap in Lambert projection coordinates to ASCII VTK")
    parser.add_argument("input_filename", help="Some input file (not used here, can be metadata)")
    args = parser.parse_args()

    input_filename = args.input_filename

    args = parser.parse_args()

    input_filename = args.input_filename

    user_inputs = read_user_input(input_filename)
    print("User inputs:", user_inputs)
    area = user_inputs.get("area", None)

    lambert_conformal = write_US_map_vtk("USMap_LambertProj.vtk", area)

