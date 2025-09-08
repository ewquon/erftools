from importlib import resources
import numpy as np
import pandas as pd
from pyproj import Transformer
import click

from erftools.utils.projection import create_lcc_mapping
from erftools.io.vtk import write_vtk_map


def create_map(coordinates, area, shift_to_origin=False):
    """Calculate Lambert conformal conic projection for the given area
    and transform the specified coordinates

    The area is described by:
        (lat_max, lon_min, lat_min, lon_max)
    """
    lambert_conformal = create_lcc_mapping(area)

    # Create transformer FROM geographic (lon/lat) TO Lambert
    transformer = Transformer.from_crs("EPSG:4326",
                                       lambert_conformal,
                                       always_xy=True)

    # region_id is optional 
    if coordinates.shape[1] == 2:
        npts = len(coordinates)
        coordinates = np.column_stack((coordinates, np.zeros(npts)))

    # Process each latitude and longitude, convert to projected coordinates
    x_trans = []
    y_trans = []
    id_vec = []
    for lon, lat, region_id in coordinates:
        x, y = transformer.transform(lon, lat)
        x_trans.append(x)
        y_trans.append(y)
        id_vec.append(int(region_id))
    x_trans = np.array(x_trans)
    y_trans = np.array(y_trans)

    # Shift coordinates to ensure minimum x and y start at 0
    if shift_to_origin:
        x_trans = x_trans - np.min(x_trans)
        y_trans = y_trans - np.min(y_trans)

    return x_trans, y_trans, id_vec

def create_US_map(area, mapfile='US_state_borders_coordinates.txt', **kwargs):
    """Create a map of the US in projected coordinates

    Coordinates are transformed with a Lambert conformal conic
    projection centered on the given area.
    """
    with resources.open_text('erftools.data', mapfile) as f:
        coords = np.loadtxt(f)
    return create_map(coords, area, **kwargs)


@click.command()
@click.argument('output_file', type=click.Path(writable=True))
@click.option('--area', nargs=4, type=float,
              help='Bounding box: lat_max, lon_min, lat_min, lon_max')
def write_US_map(output_file, area):
    """Write out a map of the United States for visualization.

    A map of the US is transformed with the Lambert conformal conic
    projection centered on the given area and written to a VTK file in
    ASCII polydata format
    """
    write_vtk_map(*create_US_map(area), output_file)


@click.command()
@click.argument('input_file', type=click.Path(exists=True, readable=True))
@click.argument('output_file', type=click.Path(writable=True))
@click.option('--area', nargs=4, type=float,
              help='Bounding box: lat_max, lon_min, lat_min, lon_max')
@click.option('--elev', type=float, default=5000.0,
              help='Constant elevation (z) for all points')
def write_map_region(input_file, output_file, area, elev):
    """Write out a bounded region for visualization.

    The bounded region (specified as a whitespace- or comma-delimited
    text file) within the specified area is transformed with the Lambert
    conformal conic projection and then written to a VTK file in ASCII
    polydata format
    """
    if input_file.lower().endswith('.csv'):
        coords = pd.read_csv(input_file).values
    else:
        coords = np.loadtxt(input_file)
    x_trans, y_trans, id_vec = create_map(coords, area)
    write_vtk_map(x_trans, y_trans, id_vec, output_file, zlo=elev,
                  point_data={"Longitude": coords[:,0],
                              "Latitude": coords[:,1]})

