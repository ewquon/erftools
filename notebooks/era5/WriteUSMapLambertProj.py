import sys
import os
import argparse
from pyproj import Transformer
import numpy as np
import pandas as pd
import pyvista as pv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from erftools.preprocessing import Download_ERA5_Data
from erftools.preprocessing import Download_ERA5_ForecastData
from erftools.preprocessing import ReadERA5_3DData

from erftools.utils.projection import create_lcc_mapping


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


def write_vtk_states(x, y, count, filename):
    """
    Write a VTK file containing borders for all states.

    Parameters:
        x (list or ndarray): List or array of x-coordinates.
        y (list or ndarray): List or array of y-coordinates.
        count (list or ndarray): List or array of state indices corresponding to each (x, y).
        filename (str): Name of the output VTK file.
    """
    x = asarray(x)
    y = asarray(y)
    count = asarray(count)

    if len(x) != len(y) or len(x) != len(count):
        raise ValueError("The length of x, y, and count must be the same.")

    # Open VTK file for writing
    with open(filename, 'w') as vtk_file:
        # Write VTK header
        vtk_file.write("# vtk DataFile Version 3.0\n")
        vtk_file.write("State borders\n")
        vtk_file.write("ASCII\n")
        vtk_file.write("DATASET POLYDATA\n")

        # Group points by state
        unique_states = unique(count)
        points = []  # List of all points
        lines = []  # List of all lines

        # Process each state
        check = 0
        for state in unique_states:
            if(check >=0):
                state_indices = where(count == state)[0]  # Indices for this state
                state_points = [(x[i], y[i]) for i in state_indices]
                start_idx = len(points)  # Starting index for this state's points
                points.extend(state_points)

                # Create line segments connecting adjacent points
                for i in range(len(state_points) - 1):
                    lines.append((start_idx + i, start_idx + i + 1))
            check = check+1;

        # Write points
        vtk_file.write(f"POINTS {len(points)} float\n")
        for px, py in points:
            vtk_file.write(f"{px} {py} 1e-12\n")

        # Write lines
        vtk_file.write(f"LINES {len(lines)} {3 * len(lines)}\n")
        for p1, p2 in lines:
            vtk_file.write(f"2 {p1} {p2}\n")


def WriteUSMapVTKFile(area):
    # Main script to process coordinates
    coordinates = np.loadtxt('StateBordersCoordinates.txt')  # Load lon, lat from a file
    utm_x = []
    utm_y = []

    lambert_conformal = create_lcc_mapping(area)

    # Create transformer FROM geographic (lon/lat) TO Lambert
    transformer = Transformer.from_crs("EPSG:4326", lambert_conformal, always_xy=True)

    # Process each latitude and longitude
    utm_x = []
    utm_y = []
    count_vec = []

    for lon, lat, count in coordinates:
        x, y = transformer.transform(lon, lat)  # Convert (lon, lat) to Lambert
        utm_x.append(x)
        utm_y.append(y)
        count_vec.append(count)

    # Write the shifted UTM coordinates to a VTK file
    write_vtk_states(utm_x, utm_y, count_vec, "USMap_LambertProj.vtk")
    return lambert_conformal


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

    lambert_conformal = WriteUSMapVTKFile(area)

