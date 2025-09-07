import os
import tqdm
import numpy as np
import pandas as pd

from erftools.preprocessing.nwpdata import NWPDataset
from erftools.preprocessing.gribdata import GribData
from erftools.constants import CONST_GRAV as const_g
from erftools.utils.microphysics import p_sat

class GFSDataset(NWPDataset):
    """NCAR Global Forecast System (GFS) global analysis data

    Extra Init Parameters
    ---------------------
    product: str or float, optional
        NCAR RDA identifier, e.g., 'd083002' or 'd084001'. This will be
        automatically chosen if not specified
    """

    def _setup(self, **kwargs):
        """Do dataset-specific setup, validation tasks"""
        # get name of RDA data product
        default_product = 'forecast' if self.forecast > 0 else 'final'
        self.product = kwargs.get('product', default_product)

        # get urls and filenames, which differ depending on the
        # analysis product
        self.datetimes, self.urls, self.filenames = construct_urls_filenames(
            self.analysis_datetime,
            self.forecast,
            self.product)

    def read(self, clip=True):
        self.grib = GribData(
            gh='Geopotential height',
            temp='Temperature',
            rh='Relative humidity',
            theta='Potential temperature',
            p='Pressure',
            u='U component of wind',
            v='V component of wind',
            w='Geometric vertical velocity',
            qv='Specific humidity',
            qc='Cloud mixing ratio',
            qr='Rain mixing ratio',
            vort='Absolute vorticity')
        print('NOTE: ONLY READ FIRST GRIBFILE FOR NOW')
        self.grib.read(self.filenames[0])
        print('NOTE: PRESSURE LEVELS BASED ON TEMP -- VERIFY THIS MAKES SENSE')
        self.pressure_levels = self.grib.pressure_levels['temp']
        if clip:
            self._clip_grid_data()

    def _clip_grid_data(self):
        # Extract unique latitude and longitude values
        unique_lats = np.unique(self.grib.lats[:, 0])  # Take the first column for unique latitudes
        unique_lons = np.unique(self.grib.lons[0, :])  # Take the first row for unique longitudes

        print("Min max lat lons are ", unique_lats[0], unique_lats[-1], unique_lons[0], unique_lons[-1]);

        nlats = len(unique_lats)
        nlons = len(unique_lons)

        lat_max = self.area[0]
        lon_min = 360.0 + self.area[1]
        lat_min = self.area[2]
        lon_max = 360.0 + self.area[3]

        print("Lat/lon min/max are ", lat_min, lat_max, lon_min, lon_max)

        # Assume regular grid
        lat_resolution = unique_lats[1] - unique_lats[0]
        lon_resolution = unique_lons[1] - unique_lons[0]

        lat_start = int((lat_min - unique_lats[0]) / lat_resolution)
        lat_end   = int((lat_max - unique_lats[0]) / lat_resolution)
        lon_start = int((lon_min - unique_lons[0]) / lon_resolution)
        lon_end   = int((lon_max - unique_lons[0]) / lon_resolution)

        domain_lats = unique_lats[lat_start:lat_end+1]
        domain_lons = unique_lons[lon_start:lon_end+1]

        print("The min max are",(lat_start, lat_end, lon_start, lon_end));

        nx = domain_lats.shape[0]
        ny = domain_lons.shape[0]

        print("nx and ny here are ", nx, ny)

        self.grib.clip(slice(nlats-lat_end-1, nlats-lat_start),
                       slice(lon_start, lon_end+1))


def construct_urls_filenames(datetime, forecast, product):
    rda_prefix = 'https://data-osdf.rda.ucar.edu/ncar/rda'

    if product.lower() in ['forecast', 'd084001', 84.1]:
        # Historical forecast data (0.25 deg x 0.25 deg grids; runs every 6h;
        # forecasts times every 3h up to 240h, every 12h up to 384h)
        assert datetime.hour % 6 == 0, 'Invalid analysis time'

        f_hrs = np.arange(0, min(forecast, 240)+1, 3)
        if forecast > 240:
            extend = np.arange(252, min(forecast, 384)+1, 12)
            f_hrs = np.concatenate((f_hrs, extend))

        filename = datetime.strftime('gfs.0p25.%Y%m%d%H.f{:03d}.grib2')
        url = datetime.strftime(f'{rda_prefix}/d084001/%Y/%Y%m%d/{filename}')

    elif product.lower() in ['fnl', 'final', 'd083003', 83.3]:
        # Final reanalaysis data (0.25 deg x 0.25 deg grids, every 6h)
        assert datetime.hour % 6 == 0, 'Invalid analysis time'

        f_hrs = np.arange(0, min(forecast, 9)+1, 3)

        filename = datetime.strftime('gdas1.fnl0p25.%Y%m%d%H.f{:02d}.grib2')
        url = datetime.strftime(f'{rda_prefix}/d083003/%Y/%Y%m/{filename}')

    datetimes = datetime + pd.to_timedelta(f_hrs, unit='h')
    urls = [url.format(fhr) for fhr in f_hrs]
    filenames = [filename.format(fhr) for fhr in f_hrs]

    return datetimes, urls, filenames
