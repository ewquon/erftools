from importlib import resources
import numpy as np
from pyproj import Transformer

from erftools.utils.projection import create_lcc_mapping
from erftools.io.vtk import write_vtk_map


def create_map(area,
               mapfile='US_state_borders_coordinates.txt',
               shift_to_origin=False):
    """Calculate Lambert conformal conic projection for the given area
    and transform the specified map (textfile with whitespace-delimited
    longitude, latitude, and state/country/region IDs)

    The area is described by:
        (lat_max, lon_min, lat_min, lon_max)
    """
    with resources.open_text('erftools.data', mapfile) as f:
        coordinates = np.loadtxt(f)

    lambert_conformal = create_lcc_mapping(area)

    # Create transformer FROM geographic (lon/lat) TO Lambert
    transformer = Transformer.from_crs("EPSG:4326",
                                       lambert_conformal,
                                       always_xy=True)

    # Process each latitude and longitude, convert to projected coordinates
    x_trans = []
    y_trans = []
    id_vec = []
    for lon, lat, state_id in coordinates:
        x, y = transformer.transform(lon, lat)
        x_trans.append(x)
        y_trans.append(y)
        id_vec.append(state_id)

    # Shift coordinates to ensure minimum x and y start at 0
    if shift_to_origin:
        x_trans = np.array(x_trans) - min(x_trans)
        y_trans = np.array(y_trans) - min(y_trans)

    return x_trans, y_trans, id_vec, lambert_conformal


def write_US_map_vtk(filename, area, **kwargs):
    """Wrapper around `create_map` in erftools.utils.map
    and `write_vtk_map` in erftools.io.vtk
    """
    x_trans, y_trans, id_vec, lcc_proj = create_map(area, **kwargs)
    write_vtk_map(x_trans, y_trans, id_vec, filename)
    return lcc_proj
