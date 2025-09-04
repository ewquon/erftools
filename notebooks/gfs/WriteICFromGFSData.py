from importlib import resources
import sys
import os
import argparse
from pyproj import Transformer
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from erftools.preprocessing import Download_GFS_Data
from erftools.preprocessing import Download_GFS_ForecastData
from erftools.preprocessing import ReadGFS_3DData
from erftools.preprocessing import ReadGFS_3DData_UVW

from erftools.utils.projection import create_lcc_mapping


def write_vtk_states(x, y, count, filename):
    """
    Write a VTK file containing borders for all states.

    Parameters:
        x (list or ndarray): List or array of x-coordinates.
        y (list or ndarray): List or array of y-coordinates.
        count (list or ndarray): List or array of state indices corresponding to each (x, y).
        filename (str): Name of the output VTK file.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    count = np.asarray(count)

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
        unique_states = np.unique(count)
        points = []  # List of all points
        lines = []  # List of all lines

        # Process each state
        check = 0
        for state in unique_states:
            if(check >=0):
                state_indices = np.where(count == state)[0]  # Indices for this state
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
    with resources.open_text('erftools.data', 'state_borders_coordinates.txt') as f:
        coordinates = np.loadtxt(f)  # Load lon, lat from a file
    utm_x = []
    utm_y = []

    #lambert_conformal = CRS.from_proj4(
    #    "+proj=lcc +lat_1=30 +lat_2=60 +lat_0=38.5 +lon_0=-97 +datum=WGS84 +units=m +no_defs")

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

    #plt.scatter(utm_x, utm_y, s=10, c='blue', label='UTM Points')
    #plt.xlabel('UTM X')
    #plt.ylabel('UTM Y')
    #plt.title('UTM Converted Points')
    #plt.legend()
    #plt.grid()
    #plt.savefig("./Images/UTM_scatter.png")
    #plt.show()

    # Shift coordinates to ensure minimum x and y start at 0
    #utm_x = array(utm_x) - min(utm_x)
    #utm_y = array(utm_y) - min(utm_y)

    # Write the shifted UTM coordinates to a VTK file
    write_vtk_states(utm_x, utm_y, count_vec, "USMap_LambertProj.vtk")
    return lambert_conformal


if __name__ == "__main__":

    if len(sys.argv) == 1:
        print("Usage: python3 WriteICFromGFSData.py <input_filename> [--do_forecast=true]")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Download and process GFS data.")
    parser.add_argument("input_filename", help="Input filename, e.g. inputs_for_Laura")
    parser.add_argument("--do_forecast", type=bool, default=False, help="Set to true to download forecast data")

    args = parser.parse_args()

    input_filename = args.input_filename
    do_forecast = args.do_forecast

    if do_forecast:
        filenames, area = Download_GFS_ForecastData(input_filename)
        lambert_conformal = WriteUSMapVTKFile(area)
        # Create the directory if it doesn't exist
        os.makedirs("Output", exist_ok=True)
        for filename in filenames:
            print(f"Processing file: {filename}")
            ReadGFS_3DData(filename, area, lambert_conformal)
        
    else:
        filename, area = Download_GFS_Data(input_filename)
        lambert_conformal = WriteUSMapVTKFile(area)
        print("Filename is ", filename)
        print(f"Processing file: {filename}")
        ReadGFS_3DData(filename, area, lambert_conformal)
        #ReadGFS_3DData_UVW(filename, area, lambert_conformal)
